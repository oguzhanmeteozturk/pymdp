// Device kernels and host launch wrappers for the CUDA neg-EFE path.
// neg_efe_cuda.cc owns the XLA FFI boundary; this TU compiles under nvcc
// and avoids XLA FFI headers (CUDA 10.2's nvcc cannot parse them).
// Supported targets: Maxwell sm_53 through Ampere sm_87; warp-shuffle paths
// gate on __CUDA_ARCH__ for ITS-safety on Volta+.
//
// Pipeline
// --------
// * B-rollout (per-(t, f)): gather qs from the prior level's per-factor
//   buffer via parent_history, contract against B. Output sized by H_f^t.
//   When wB is supplied (param-info-gain) the same block also reduces a
//   factor_score; when v_full is supplied (use_inductive), an inductive
//   score the same way.
// * Modality scoring (per-(t, m)): over the dep factors' Cartesian history
//   product, not the joint node space.
//     - Rank 1: one thread per (b, h).
//     - Rank 2/3 small: fused-tiny kernel inlines the s-Kronecker per (b, h).
//     - Rank 2 large: q01_outer → cuBLAS GEMM → finish kernel.
//     - Rank 3 large: split off deps[rank-1]. Stage 1 builds q01_outer + tmp
//       tensors; stage 2 folds in entropy + linear.
//   deps order is the caller's A_dependencies order — same indexing in every
//   consumer, not a correctness contract. For best rank-3 perf put the
//   largest-history / smallest-state factor last (S_split sizes the
//   tmp buffers).
// * Linear: -HA + sum_o A[o,k]*C[o] is precomputed per-(t, m, b, k) so
//   utility + static entropy correction land in one K-pass dot. The dynamic
//   xlogx(qo) entropy stays at runtime.
// * Outputs: scores_concat (per-(t, m) modality), inductive_concat and
//   factor_scores (both per-(t, f)). Final scatter combines all three via
//   per-policy history-tuple lookups.

#include "neg_efe_cuda_kernels.h"

#include <cstdint>

namespace pymdp_ffi {
namespace cuda_kernels {

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------

// Block-size tiering. Maxwell sm_53 keeps the original 128; sm_87 gets 256
// for 1-D scoring + scatter kernels, where one-thread-per-
// output with short serial loops benefits from the additional latency
// hiding the bigger SM scheduler offers. B-rollout stays at 128 on both —
// it uses dynamic shared memory + per-block reductions whose unroll
// schedules were tuned around that block size; widening would require a
// different reduction tree.
constexpr int kBlockSizeDefault  = 128;
constexpr int kBlockSizeAmpere1D = 256;
constexpr int kBRolloutBlockSize = 128;

__host__ __device__ __forceinline__ int64_t idiv_ceil64(int64_t a, int64_t b) {
  return (a + b - 1) / b;
}

// total is int64 so callers can hand in raw products; CUDA enforces the grid
// dim ceiling at launch. The default block_size lets older callers / manual
// launches with no arch tiering continue to compile unchanged.
inline int launch_blocks(int64_t total, int block_size = kBlockSizeDefault) {
  return static_cast<int>(idiv_ceil64(total, static_cast<int64_t>(block_size)));
}

// Cached compute-capability lookup. cudaGetDeviceProperties is ~milliseconds
// the first time; thread_local cache means the cost amortizes to one query
// per worker thread for the process lifetime. Returns 0 on error so callers
// fall back to kBlockSizeDefault rather than masquerading as a newer arch.
inline int device_cc_cached() {
  thread_local int cc = -1;
  if (cc < 0) {
    int            dev = 0;
    cudaDeviceProp prop{};
    if (cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&prop, dev) == cudaSuccess) {
      cc = prop.major * 10 + prop.minor;
    } else {
      cc = 0;
    }
  }
  return cc;
}

// 1-D scoring/scatter kernels: 256 on sm_87+ when total work warrants it,
// else 128. Tiny shapes (e.g. agent_step Bn*P=96) leave a 256-block mostly
// empty and lose to a narrower block the scheduler can ping faster; the
// threshold ensures ~8 full 256-blocks before switching up.
//
// Stream arg is unused today but kept for a future per-stream multi-device path.
constexpr int64_t kAmpereWideMinTotal = 2048;

inline int launch_block_size_1d(cudaStream_t /*stream*/, int64_t total) {
  if (device_cc_cached() < 87) return kBlockSizeDefault;
  return (total >= kAmpereWideMinTotal) ? kBlockSizeAmpere1D : kBlockSizeDefault;
}

// 1-D grid launch macro. `total` must already be int64 at the call site —
// int32 products like `Bn * H_0 * H_1 * H_d` overflow silently and
// short-circuit the empty-grid guard as "success".
//
// Block size is selected via launch_block_size_1d(stream) — 256 on sm_87+,
// 128 elsewhere. The body must reference `pymdp_block_size` (not the now-
// removed `kBlockSize` constant) in its <<<..., pymdp_block_size, ...>>>
// launch-config and the enclosing function must expose a `stream` symbol.
//
// IMPORTANT: this macro `return`s from the enclosing function (cudaSuccess
// on empty grid, else cudaGetLastError() after launch), so it must be the
// terminal statement of a cudaError_t-returning launcher.
#define PYMDP_LAUNCH_1D(total, ...)                                                                                    \
  do {                                                                                                                 \
    const int64_t pymdp_launch_total = (total);                                                                        \
    if (pymdp_launch_total <= 0) return cudaSuccess;                                                                   \
    const int pymdp_block_size    = launch_block_size_1d(stream, pymdp_launch_total);                                  \
    const int pymdp_launch_blocks = launch_blocks(pymdp_launch_total, pymdp_block_size);                               \
    __VA_ARGS__;                                                                                                       \
    return cudaGetLastError();                                                                                         \
  } while (0)

// 1..8 switch. BODY(N) expands to `case N: ... break;`; the `default:` arm
// returns cudaErrorInvalidValue, so the macro must appear inside a
// cudaError_t-returning function.
#define PYMDP_CUDA_DISPATCH_1_8(expr, BODY)                                                                            \
  switch (expr) {                                                                                                      \
    BODY(1)                                                                                                            \
    BODY(2) BODY(3) BODY(4) BODY(5) BODY(6) BODY(7) BODY(8) default : return cudaErrorInvalidValue;                    \
  }

// Lift (use_states, use_linear, use_pA) into compile-time template params via
// an 8-way host switch — nvcc DCEs the dead arms, drops backing registers,
// and unrolls inner loops more aggressively. BODY(S, L, P) treats S/L/P as
// constexpr bool tokens.
#define PYMDP_DISPATCH_BOOL3(use_s, use_l, use_p, BODY)                                                                \
  do {                                                                                                                 \
    const int pymdp_b3 = (static_cast<int>(use_s) << 2) | (static_cast<int>(use_l) << 1) | static_cast<int>(use_p);    \
    switch (pymdp_b3) {                                                                                                \
    case 0:                                                                                                            \
      BODY(false, false, false);                                                                                       \
      break;                                                                                                           \
    case 1:                                                                                                            \
      BODY(false, false, true);                                                                                        \
      break;                                                                                                           \
    case 2:                                                                                                            \
      BODY(false, true, false);                                                                                        \
      break;                                                                                                           \
    case 3:                                                                                                            \
      BODY(false, true, true);                                                                                         \
      break;                                                                                                           \
    case 4:                                                                                                            \
      BODY(true, false, false);                                                                                        \
      break;                                                                                                           \
    case 5:                                                                                                            \
      BODY(true, false, true);                                                                                         \
      break;                                                                                                           \
    case 6:                                                                                                            \
      BODY(true, true, false);                                                                                         \
      break;                                                                                                           \
    case 7:                                                                                                            \
      BODY(true, true, true);                                                                                          \
      break;                                                                                                           \
    }                                                                                                                  \
  } while (0)

// Two-bool variant for kernels with only two runtime flags.
#define PYMDP_DISPATCH_BOOL2(use_a, use_b, BODY)                                                                       \
  do {                                                                                                                 \
    const int pymdp_b2 = (static_cast<int>(use_a) << 1) | static_cast<int>(use_b);                                     \
    switch (pymdp_b2) {                                                                                                \
    case 0:                                                                                                            \
      BODY(false, false);                                                                                              \
      break;                                                                                                           \
    case 1:                                                                                                            \
      BODY(false, true);                                                                                               \
      break;                                                                                                           \
    case 2:                                                                                                            \
      BODY(true, false);                                                                                               \
      break;                                                                                                           \
    case 3:                                                                                                            \
      BODY(true, true);                                                                                                \
      break;                                                                                                           \
    }                                                                                                                  \
  } while (0)

// Combined empty-grid guard + block count + 3-bool template dispatch +
// cudaGetLastError() return. `kernel` is the templated kernel symbol;
// `stream` is read from the calling scope. Block size is selected per-call
// via launch_block_size_1d (256 on sm_87+, 128 elsewhere).
//
// Must be the terminal statement of a cudaError_t-returning launcher.
#define PYMDP_LAUNCH_BOOL3_1D(total64, use_s, use_l, use_p, kernel, ...)                                               \
  do {                                                                                                                 \
    const int64_t pymdp_total = (total64);                                                                             \
    if (pymdp_total <= 0) return cudaSuccess;                                                                          \
    const int pymdp_block_size = launch_block_size_1d(stream, pymdp_total);                                            \
    const int pymdp_blocks     = launch_blocks(pymdp_total, pymdp_block_size);                                         \
    const int pymdp_b3 =                                                                                               \
        (static_cast<int>(use_s) << 2) | (static_cast<int>(use_l) << 1) | static_cast<int>(use_p);                     \
    switch (pymdp_b3) {                                                                                                \
    case 0: kernel<false, false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 1: kernel<false, false, true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 2: kernel<false, true , false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 3: kernel<false, true , true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 4: kernel<true , false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 5: kernel<true , false, true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 6: kernel<true , true , false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    case 7: kernel<true , true , true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;            \
    }                                                                                                                  \
    return cudaGetLastError();                                                                                         \
  } while (0)

// Two-bool variant of PYMDP_LAUNCH_BOOL3_1D.
#define PYMDP_LAUNCH_BOOL2_1D(total64, use_a, use_b, kernel, ...)                                                      \
  do {                                                                                                                 \
    const int64_t pymdp_total = (total64);                                                                             \
    if (pymdp_total <= 0) return cudaSuccess;                                                                          \
    const int pymdp_block_size = launch_block_size_1d(stream, pymdp_total);                                            \
    const int pymdp_blocks     = launch_blocks(pymdp_total, pymdp_block_size);                                         \
    const int pymdp_b2         = (static_cast<int>(use_a) << 1) | static_cast<int>(use_b);                             \
    switch (pymdp_b2) {                                                                                                \
    case 0: kernel<false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;                   \
    case 1: kernel<false, true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;                   \
    case 2: kernel<true , false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;                   \
    case 3: kernel<true , true ><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__); break;                   \
    }                                                                                                                  \
    return cudaGetLastError();                                                                                         \
  } while (0)

// -----------------------------------------------------------------------------
// B-rollout — factor-local or multi-parent B-deps.
//
//   qs_outer[b, h, k] = prod_i parents.qs[i][b, parent_h[h, i], s_i(k)]
//   qs_out  [b, h, s] = sum_k B[b, s, k, action[h]] * qs_outer[b, h, k]
// K_f = prod_i parents.S[i]; k decomposes (s_0, ..., s_{N-1}) row-major.
// One block per (b, h_next); shmem holds qs_outer[K_f].
//
// Shmem ceiling: K_f * 4B per block. sm_53 has 48 KB/SM so K_f <= 12288;
// larger trips cudaErrorInvalidConfiguration (chunk the K loop).
//
// N_PARENTS specialization is compile-time: N in {1,2,3} hoists per-parent
// qs/H/S into named __restrict__ locals; N >= 4 takes an indexed Kronecker
// walk via BRolloutParents. Adding a hand-tuned N=4 arm = one new else-if.
// -----------------------------------------------------------------------------

// Base pointer of parent factor i's qs[b, parent_h_i, :] within the parent
// buffer. Phase 1 of B-rollout reads this for each parent.
__device__ __forceinline__ const float* parent_qs_base(const BRolloutParents& p, int i, int b, int parent_h_i) {
  return p.qs[i] + (static_cast<size_t>(b) * p.H[i] + parent_h_i) * p.S[i];
}

// Warp-shuffle sum over the 32 lanes of warp 0. Pre-Volta uses legacy
// __shfl_xor; Volta+ requires the _sync variant for ITS correctness.
// Precondition: all 32 lanes enter uniformly (the `if (tid < 32)` guard
// at the call site).
__device__ __forceinline__ float warp0_shfl_sum(float v) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
#else
  v += __shfl_xor(v, 16);
  v += __shfl_xor(v, 8);
  v += __shfl_xor(v, 4);
  v += __shfl_xor(v, 2);
  v += __shfl_xor(v, 1);
#endif
  return v;
}

// Warp-wide argmax: each lane brings (val, idx); all lanes converge on the
// pair with maximum val. Ties broken by smaller idx so the result is
// deterministic across runs.
__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 700)
    const float other_val = __shfl_xor_sync(0xffffffffu, val, off);
    const int   other_idx = __shfl_xor_sync(0xffffffffu, idx, off);
#else
    const float other_val = __shfl_xor(val, off);
    const int   other_idx = __shfl_xor(idx, off);
#endif
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// Block-wide sum, parameterized on BLOCK_SIZE. Warp-leader pattern: each warp
// reduces internally via shuffle (no shmem), warp leaders publish to a
// per-warp shmem slot, single __syncthreads, then warp 0 shuffles across
// the warp-leader slots. ONE barrier per call vs. the prior tree's 2-3 at
// BLOCK_SIZE >= 128 — the dominant cost in B-rollout's profile was
// __syncthreads in this reducer. Shmem footprint drops from BLOCK_SIZE
// floats to BLOCK_SIZE/32 floats.
//
// CONTRACT — read before adding a caller:
//   1. The return value is valid **only on lane 0** (threadIdx.x == 0). The
//      `_lane0` suffix flags this at every call site. If you need the value
//      broadcast to all threads, publish it yourself through shmem + a
//      barrier.
//   2. The static __shared__ `buf` is aliased across inlined call sites.
//      Back-to-back invocations race on `buf` (warp 0 from call #1 still
//      reading buf[0..num_warps-1] while warp leaders overwrite it for
//      call #2). Callers that invoke this helper twice in a row must place
//      a __syncthreads() between the calls — or use the pair variant below
//      to fuse two reductions into one barrier.
template <int BLOCK_SIZE>
__device__ __forceinline__ float block_reduce_sum_lane0(float val) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float buf[NUM_WARPS];

  // Intra-warp reduction via XOR butterfly. After this, lane 0 of each warp
  // holds that warp's partial sum.
  val = warp0_shfl_sum(val);

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (lane == 0) buf[warp] = val;
  __syncthreads();

  // Warp 0 reduces the NUM_WARPS partials. Lanes past NUM_WARPS contribute
  // identity (0); the XOR butterfly converges on lane 0 in log2(32) steps.
  float v = 0.0f;
  if (warp == 0) {
    v = (lane < NUM_WARPS) ? buf[lane] : 0.0f;
    v = warp0_shfl_sum(v);
  }
  return v;  // Valid only on lane 0 — see contract above.
}

// Fused two-reduction variant. Reduces a pair (a, b) using a single
// __syncthreads instead of three (sync + reduce, sync between calls, sync +
// reduce). Used by B-rollout when both COMPUTE_WB and COMPUTE_INDUCTIVE are
// live so the per-block factor_score + ind_score pair publishes with one
// barrier rather than the back-to-back calls that the buf-aliasing contract
// of the single-value helper would otherwise force.
//
// Same contract: results are valid on lane 0 only.
template <int BLOCK_SIZE>
__device__ __forceinline__ void block_reduce_sum_pair_lane0(float& a, float& b) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");
  constexpr int NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float buf_a[NUM_WARPS];
  __shared__ float buf_b[NUM_WARPS];

  a = warp0_shfl_sum(a);
  b = warp0_shfl_sum(b);

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

  if (lane == 0) {
    buf_a[warp] = a;
    buf_b[warp] = b;
  }
  __syncthreads();

  if (warp == 0) {
    float va = (lane < NUM_WARPS) ? buf_a[lane] : 0.0f;
    float vb = (lane < NUM_WARPS) ? buf_b[lane] : 0.0f;
    va       = warp0_shfl_sum(va);
    vb       = warp0_shfl_sum(vb);
    a        = va;
    b        = vb;
  }
}

// Phase 2 of B-rollout (per-s dot against B → qs_out), optionally fused with
//   factor_score = sum_s (qs_out[s] * sum_k wB[s,k] * qs_outer[k])         (wB)
//   ind_score    = sum_s (qs_out[s] * v_full_b[s])                  (inductive)
// Both reductions share the per-block reduce. COMPUTE_WB / COMPUTE_INDUCTIVE
// are template params so the no-reduction instantiation DCEs the reduce
// buffer and partial registers.
template <int BLOCK_SIZE, bool COMPUTE_WB, bool COMPUTE_INDUCTIVE>
__device__ __forceinline__ void
b_rollout_phase2_and_wb(const float* __restrict__ B, const float* __restrict__ wB_tr,
                        const float* __restrict__ v_full_b, int b, int h, int u, int Nh, int S_f, int K_f, int U_f,
                        const float* __restrict__ qs_outer, float* __restrict__ qs_out,
                        float* __restrict__ factor_score, float* __restrict__ ind_score_t_f, int64_t ind_b_stride) {
  // Per-block bases hoisted in size_t; inner loops then run int32.
  const int    row_stride_B = K_f * U_f;
  const float* B_b          = B + static_cast<size_t>(b) * S_f * row_stride_B;
  const size_t score_off    = static_cast<size_t>(b) * Nh + h;
  float* const qs_out_bh    = qs_out + score_off * S_f;

  const float* wb_bu = COMPUTE_WB ? wB_tr + (static_cast<size_t>(b) * U_f + u) * S_f * K_f : nullptr;

  // Threads with no s assigned contribute 0; threads owning multiple s-stripes
  // accumulate over them. B[b, s, k, u] = b*S_f*K_f*U_f + s*K_f*U_f + k*U_f + u.
  // Stride by BLOCK_SIZE (constexpr) rather than blockDim.x so the inner s-loop
  // unrolls / bounds-check elides; the kernel launch must use the matching block.
  float partial     = 0.0f;
  float partial_ind = 0.0f;
  for (int s = threadIdx.x; s < S_f; s += BLOCK_SIZE) {
    const float* Bf   = B_b + s * row_stride_B;
    const float* wb_s = COMPUTE_WB ? (wb_bu + s * K_f) : nullptr;
    float        acc  = 0.0f;
    float        wacc = 0.0f;
    // Fused k-loop shares the qs_outer[k] load between B and wB when wB is live.
    for (int k = 0; k < K_f; ++k) {
      const float qok = qs_outer[k];
      acc += Bf[k * U_f + u] * qok;
      if (COMPUTE_WB) wacc += wb_s[k] * qok;
    }
    qs_out_bh[s] = acc;
    if (COMPUTE_WB) partial += acc * wacc;
    if (COMPUTE_INDUCTIVE) partial_ind += acc * v_full_b[s];
  }

  // No barrier before the reductions — partials are per-thread registers; the
  // kernel-exit barrier handles qs_out visibility downstream.
  //
  // factor_score / ind_score_t_f are per-(t, f) slices of [Bn, total_ind]
  // buffers — per-batch stride is ind_b_stride, NOT Nh. qs_out uses its own
  // [Bn, Nh, S_f] layout above (different buffer, different stride).
  //
  // WB+IND case fuses the two reductions into one barrier via the pair
  // helper; the single-reduction instantiations DCE the unused branch.
  const int64_t score_b_h = static_cast<int64_t>(b) * ind_b_stride + h;
  if (COMPUTE_WB && COMPUTE_INDUCTIVE) {
    block_reduce_sum_pair_lane0<BLOCK_SIZE>(partial, partial_ind);
    if (threadIdx.x == 0) {
      factor_score[score_b_h]  = partial;
      ind_score_t_f[score_b_h] = partial_ind;
    }
  } else if (COMPUTE_WB) {
    const float total_lane0 = block_reduce_sum_lane0<BLOCK_SIZE>(partial);
    if (threadIdx.x == 0) factor_score[score_b_h] = total_lane0;
  } else if (COMPUTE_INDUCTIVE) {
    const float total_ind_lane0 = block_reduce_sum_lane0<BLOCK_SIZE>(partial_ind);
    if (threadIdx.x == 0) ind_score_t_f[score_b_h] = total_ind_lane0;
  }
}

template <int BLOCK_SIZE, int N_PARENTS, bool COMPUTE_WB, bool COMPUTE_INDUCTIVE>
__global__ void b_rollout_general_kernel(const float* __restrict__ B, const float* __restrict__ wB_tr,
                                         const float* __restrict__ v_full, int qs_flat, int qs_off_f,
                                         const int32_t* __restrict__ action_h,
                                         const int32_t* __restrict__ parent_histories, BRolloutParents parents, int Bn,
                                         int Nh, int S_f, int K_f, int U_f, float* __restrict__ qs_out,
                                         float* __restrict__ factor_score, float* __restrict__ ind_score_t_f,
                                         int64_t ind_b_stride) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  if (b >= Bn || h >= Nh) return;

  extern __shared__ float qs_outer[];

  const int u = action_h[h];

  // Phase 1: qs_outer[K_f] = prod_i parents.qs[i][b, parent_h_i, s_i(k)].
  // The N_PARENTS cascade DCEs to one arm per instantiation.
  if (N_PARENTS == 1) {
    const float* __restrict__ qs0_bh = parent_qs_base(parents, 0, b, parent_histories[h]);
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) qs_outer[k] = qs0_bh[k];
  } else if (N_PARENTS == 2) {
    const int                 S1     = parents.S[1];
    const float* __restrict__ qs0_bh = parent_qs_base(parents, 0, b, parent_histories[h * 2 + 0]);
    const float* __restrict__ qs1_bh = parent_qs_base(parents, 1, b, parent_histories[h * 2 + 1]);
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      const int s_0 = k / S1;
      const int s_1 = k - s_0 * S1;
      qs_outer[k]   = qs0_bh[s_0] * qs1_bh[s_1];
    }
  } else if (N_PARENTS == 3) {
    const int                 S1     = parents.S[1];
    const int                 S2     = parents.S[2];
    const int                 S_12   = S1 * S2;
    const float* __restrict__ qs0_bh = parent_qs_base(parents, 0, b, parent_histories[h * 3 + 0]);
    const float* __restrict__ qs1_bh = parent_qs_base(parents, 1, b, parent_histories[h * 3 + 1]);
    const float* __restrict__ qs2_bh = parent_qs_base(parents, 2, b, parent_histories[h * 3 + 2]);
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      const int s_0 = k / S_12;
      int       rem = k - s_0 * S_12;
      const int s_1 = rem / S2;
      const int s_2 = rem - s_1 * S2;
      qs_outer[k]   = qs0_bh[s_0] * qs1_bh[s_1] * qs2_bh[s_2];
    }
  } else {
    // Generic N_PARENTS >= 4. Indexed Kronecker walk via the parents struct;
    // the per-parent base uses size_t widening just like the explicit arms.
    int parent_h_local[N_PARENTS];
#pragma unroll
    for (int i = 0; i < N_PARENTS; ++i) parent_h_local[i] = parent_histories[h * N_PARENTS + i];
    int strides[N_PARENTS];
    strides[N_PARENTS - 1] = 1;
#pragma unroll
    for (int i = N_PARENTS - 2; i >= 0; --i) strides[i] = strides[i + 1] * parents.S[i + 1];
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      int   rem = k;
      float val = 1.0f;
#pragma unroll
      for (int i = 0; i < N_PARENTS; ++i) {
        const int s_i = (i + 1 < N_PARENTS) ? (rem / strides[i]) : rem;
        rem -= s_i * strides[i];
        val *= parent_qs_base(parents, i, b, parent_h_local[i])[s_i];
      }
      qs_outer[k] = val;
    }
  }
  __syncthreads();

  const float* v_full_b =
      COMPUTE_INDUCTIVE ? (v_full + static_cast<size_t>(b) * qs_flat + qs_off_f) : nullptr;
  b_rollout_phase2_and_wb<BLOCK_SIZE, COMPUTE_WB, COMPUTE_INDUCTIVE>(B, wB_tr, v_full_b, b, h, u, Nh, S_f, K_f, U_f,
                                                                    qs_outer, qs_out, factor_score, ind_score_t_f,
                                                                    ind_b_stride);
}

namespace {

// Architectural compatibility: per-block static shared memory is capped at
// 48 KB on every CUDA arch. Above 48 KB you must (a) request *dynamic*
// shared memory at launch and (b) explicitly raise this kernel's per-launch
// dynamic shared-memory ceiling via cudaFuncSetAttribute. We also flip
// PreferredSharedMemoryCarveout to cudaSharedmemCarveoutMaxShared so the SM
// reserves shared in favor of L1 — qs_outer is the whole point of the
// dynamic allocation, so trading L1 for shared is a free win on this
// kernel. Both `cudaFuncSetAttribute` and the two relevant attribute
// enums are present in CUDA 9.0+, so Maxwell sm_53 / nvcc 10.2 stays
// compatible (the helper short-circuits below 48 KB, which is the entire
// Maxwell envelope).
//
// Per-instantiation thread_local cache: cudaFuncSetAttribute is persistent
// (the attribute sticks for the kernel symbol after one successful call),
// so we record the largest size we've successfully raised the kernel to
// and skip re-querying. cudaFuncGetAttributes is several hundred
// nanoseconds and cudaFuncSetAttribute slightly more — not catastrophic
// per launch, but avoidable on steady-state.
template <int BLOCK_SIZE, int N_PARENTS, bool COMPUTE_WB, bool COMPUTE_INDUCTIVE>
inline cudaError_t configure_b_rollout_for_arch(size_t shmem_bytes) {
  if (shmem_bytes <= 48u * 1024u) return cudaSuccess;
  thread_local size_t configured_max = 0;
  if (shmem_bytes <= configured_max) return cudaSuccess;
  auto kernel = b_rollout_general_kernel<BLOCK_SIZE, N_PARENTS, COMPUTE_WB, COMPUTE_INDUCTIVE>;
  cudaFuncAttributes attr{};
  cudaError_t        err = cudaFuncGetAttributes(&attr, kernel);
  if (err != cudaSuccess) return err;
  if (shmem_bytes > static_cast<size_t>(attr.maxDynamicSharedSizeBytes)) {
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shmem_bytes));
    if (err != cudaSuccess) return err;
    err = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    if (err != cudaSuccess) return err;
  }
  configured_max = shmem_bytes;
  return cudaSuccess;
}

template <int N_PARENTS, bool COMPUTE_WB, bool COMPUTE_INDUCTIVE>
inline cudaError_t launch_b_rollout_tpl(const float* B, const float* wB_tr, const float* v_full, int qs_flat,
                                        int qs_off_f, const int32_t* action_h, const int32_t* parent_histories,
                                        BRolloutParents parents, int Bn, int Nh, int S_f, int K_f, int U_f,
                                        float* qs_out, float* factor_score, float* ind_score_t_f,
                                        int64_t ind_b_stride, cudaStream_t stream) {
  // dim3(0, ...) trips cudaErrorInvalidConfiguration on some drivers.
  if (Bn <= 0 || Nh <= 0) return cudaSuccess;
  const dim3   grid(static_cast<unsigned>(Bn), static_cast<unsigned>(Nh), 1);
  const size_t shmem_bytes = static_cast<size_t>(K_f) * sizeof(float);
  cudaError_t  err         = configure_b_rollout_for_arch<kBRolloutBlockSize, N_PARENTS, COMPUTE_WB,
                                                          COMPUTE_INDUCTIVE>(shmem_bytes);
  if (err != cudaSuccess) return err;
  b_rollout_general_kernel<kBRolloutBlockSize, N_PARENTS, COMPUTE_WB, COMPUTE_INDUCTIVE>
      <<<grid, kBRolloutBlockSize, shmem_bytes, stream>>>(B, wB_tr, v_full, qs_flat, qs_off_f, action_h,
                                                          parent_histories, parents, Bn, Nh, S_f, K_f, U_f, qs_out,
                                                          factor_score, ind_score_t_f, ind_b_stride);
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_b_rollout_general(const float* B, const float* wB_tr, const float* v_full, int qs_flat, int qs_off_f,
                                     const int32_t* action_h, const int32_t* parent_histories,
                                     const BRolloutParents& parents, int n_parents, int Bn, int Nh, int S_f, int K_f,
                                     int U_f, float* qs_out, float* factor_score, float* ind_score_t_f,
                                     int64_t ind_b_stride, cudaStream_t stream) {
  // Empty-grid guard lives in the public wrapper, not the inner _tpl: a no-op
  // dispatch path would otherwise leave the trailing cudaGetLastError() free
  // to surface a stale prior-launch error.
  if (Bn <= 0 || Nh <= 0) return cudaSuccess;
  // 8 N_PARENTS x 4 bool combos = 32 instantiations, dead reduction arms DCE'd.
  const bool compute_wb        = (wB_tr != nullptr && factor_score != nullptr);
  const bool compute_inductive = (v_full != nullptr && ind_score_t_f != nullptr);
  // _tpl now returns cudaError_t (configure_b_rollout_for_arch may fail
  // before the launch even runs); propagate that instead of relying on the
  // post-dispatch cudaGetLastError(), which would only catch launch errors.
  cudaError_t pymdp_br_rc = cudaSuccess;
#define LAUNCH_B_ROLLOUT_BOOL(W, I)                                                                                    \
  pymdp_br_rc = launch_b_rollout_tpl<PYMDP_BR_N_VAL, W, I>(                                                            \
      B, wB_tr, v_full, qs_flat, qs_off_f, action_h, parent_histories, parents, Bn, Nh, S_f, K_f, U_f, qs_out,         \
      factor_score, ind_score_t_f, ind_b_stride, stream)
#define LAUNCH_B_ROLLOUT(N)                                                                                            \
  case N: {                                                                                                            \
    constexpr int PYMDP_BR_N_VAL = N;                                                                                  \
    PYMDP_DISPATCH_BOOL2(compute_wb, compute_inductive, LAUNCH_B_ROLLOUT_BOOL);                                        \
    break;                                                                                                             \
  }
  PYMDP_CUDA_DISPATCH_1_8(n_parents, LAUNCH_B_ROLLOUT)
#undef LAUNCH_B_ROLLOUT
#undef LAUNCH_B_ROLLOUT_BOOL
  return (pymdp_br_rc != cudaSuccess) ? pymdp_br_rc : cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Common scoring helpers
// -----------------------------------------------------------------------------

// xlogx clamp matching the CPU kLogEps.
__device__ __forceinline__ float clamp_log(float v) {
  v = fmaxf(v, 1e-16f);
  return v * __logf(v);
}

template <bool USE_STATES>
__device__ __forceinline__ void write_modality_score(float* __restrict__ score_out, int64_t b, int64_t b_stride,
                                                     int64_t local_h, float acc_ent, float linear_acc) {
  score_out[b * b_stride + local_h] = (USE_STATES ? -acc_ent : 0.0f) + linear_acc;
}

// -----------------------------------------------------------------------------
// Rank-1 modality scoring. One thread per (b, h_d_0).
//   qo[b, h, o] = sum_{s_0} A[b, o, s_0] * qs[b, h, s_0]
//   score[b, h] = (-sum_o xlogx(qo))                 if use_states
//               + (sum_{s_0} linear * qs)            if use_linear
//               + (sum_o qo * sum_{s_0} wA * qs)     if use_pA
// -----------------------------------------------------------------------------

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank1_kernel(const float* __restrict__ A, const float* __restrict__ wA,
                                                  const float* __restrict__ linear, const float* __restrict__ qs_d_0,
                                                  int Bn, int O, int H_d_0, int S_d_0, int64_t b_stride,
                                                  float* __restrict__ score_out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * H_d_0;
  if (idx >= total) return;
  const int b = static_cast<int>(idx / H_d_0);
  const int h = static_cast<int>(idx % H_d_0);

  const float* A_b      = A + b * O * S_d_0;
  const float* wA_b     = USE_PA ? (wA + b * O * S_d_0) : nullptr;
  const float* qs_p     = qs_d_0 + (b * H_d_0 + h) * S_d_0;
  const float* linear_b = USE_LINEAR ? (linear + b * S_d_0) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
    // Fused A/wA inner: one qs_p[i] load feeds both contractions.
    for (int o = 0; o < O; ++o) {
      const float* A_bo  = A_b + o * S_d_0;
      const float* wA_bo = USE_PA ? (wA_b + o * S_d_0) : nullptr;
      float        qo_o  = 0.0f;
      float        wa_o  = 0.0f;
      for (int i = 0; i < S_d_0; ++i) {
        const float q = qs_p[i];
        qo_o += A_bo[i] * q;
        if (USE_PA) wa_o += wA_bo[i] * q;
      }
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) pA_acc += qo_o * wa_o;
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int i = 0; i < S_d_0; ++i) linear_acc += linear_b[i] * qs_p[i];
  }

  // score_out is the (t, m) slice of scores_concat [Bn, total_mod_entries];
  // b-stride = total_mod_entries (same convention for all rank-N kernels).
  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

cudaError_t launch_modality_score_dedup_rank1(const float* A_unflat, const float* wA_unflat, const float* linear,
                                              const float* qs_d_0, int Bn, int O, int H_d_0, int S_d_0,
                                              int64_t b_stride, bool use_states, bool use_linear, bool use_pA,
                                              float* score_out, cudaStream_t stream) {
  PYMDP_LAUNCH_BOOL3_1D(static_cast<int64_t>(Bn) * H_d_0, use_states, use_linear, use_pA,
                        modality_score_dedup_rank1_kernel,
                        A_unflat, wA_unflat, linear, qs_d_0, Bn, O, H_d_0, S_d_0, b_stride, score_out);
}

// -----------------------------------------------------------------------------
// Tiny-shape fused rank-2 / rank-3 scoring. Inline s-Kronecker per (b, h);
// no q01_outer / no cuBLAS / no tmp buffers. Gated by the host heuristic
// `use_tiny_fused_modality_path` (in neg_efe_cuda.cc); larger shapes fall
// through to the cuBLAS pipeline below.
// -----------------------------------------------------------------------------

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_fused_tiny_kernel(const float* __restrict__ A, const float* __restrict__ wA,
                                             const float* __restrict__ linear, const float* __restrict__ qs_d_0,
                                             const float* __restrict__ qs_d_1, int Bn, int O, int H_d_0, int H_d_1,
                                             int S_d_0, int S_d_1, int64_t b_stride, float* __restrict__ score_out) {
  const int64_t H_kk  = static_cast<int64_t>(H_d_0) * H_d_1;
  const int64_t K_d   = static_cast<int64_t>(S_d_0) * S_d_1;
  const int64_t total = static_cast<int64_t>(Bn) * H_kk;

  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int b  = static_cast<int>(idx / H_kk);
  const int h  = static_cast<int>(idx - static_cast<int64_t>(b) * H_kk);
  const int h0 = h / H_d_1;
  const int h1 = h - h0 * H_d_1;

  const float* qs0   = qs_d_0 + (b * H_d_0 + h0) * S_d_0;
  const float* qs1   = qs_d_1 + (b * H_d_1 + h1) * S_d_1;
  const float* A_b   = A + static_cast<size_t>(b) * O * K_d;
  const float* wA_b  = USE_PA ? (wA + static_cast<size_t>(b) * O * K_d) : nullptr;
  const float* lin_b = USE_LINEAR ? (linear + static_cast<size_t>(b) * K_d) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;

  if (USE_STATES || USE_PA) {
    for (int o = 0; o < O; ++o) {
      const float* A_bo  = A_b + o * K_d;
      const float* wA_bo = USE_PA ? (wA_b + o * K_d) : nullptr;

      float qo_o = 0.0f;
      float wa_o = 0.0f;

      int k = 0;
      for (int s0 = 0; s0 < S_d_0; ++s0) {
        const float q0 = qs0[s0];
        for (int s1 = 0; s1 < S_d_1; ++s1, ++k) {
          const float q = q0 * qs1[s1];
          qo_o += A_bo[k] * q;
          if (USE_PA) wa_o += wA_bo[k] * q;
        }
      }

      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) pA_acc += qo_o * wa_o;
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    int k = 0;
    for (int s0 = 0; s0 < S_d_0; ++s0) {
      const float q0 = qs0[s0];
      for (int s1 = 0; s1 < S_d_1; ++s1, ++k) {
        linear_acc += lin_b[k] * q0 * qs1[s1];
      }
    }
  }

  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

cudaError_t launch_modality_score_dedup_rank2_fused_tiny(const float* A_unflat, const float* wA_unflat,
                                                         const float* linear, const float* qs_d_0, const float* qs_d_1,
                                                         int Bn, int O, int H_d_0, int H_d_1, int S_d_0, int S_d_1,
                                                         int64_t b_stride, bool use_states, bool use_linear,
                                                         bool use_pA, float* score_out, cudaStream_t stream) {
  PYMDP_LAUNCH_BOOL3_1D(static_cast<int64_t>(Bn) * H_d_0 * H_d_1, use_states, use_linear, use_pA,
                        modality_score_dedup_rank2_fused_tiny_kernel,
                        A_unflat, wA_unflat, linear, qs_d_0, qs_d_1, Bn, O, H_d_0, H_d_1, S_d_0, S_d_1, b_stride,
                        score_out);
}

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank3_fused_tiny_kernel(
    const float* __restrict__ A, const float* __restrict__ wA, const float* __restrict__ linear,
    const float* __restrict__ qs_d_0, const float* __restrict__ qs_d_1, const float* __restrict__ qs_d_2, int Bn, int O,
    int H_d_0, int H_d_1, int H_d_2, int S_d_0, int S_d_1, int S_d_2, int64_t b_stride, float* __restrict__ score_out) {
  const int64_t H_12  = static_cast<int64_t>(H_d_1) * H_d_2;
  const int64_t H_all = static_cast<int64_t>(H_d_0) * H_12;
  const int64_t K_d   = static_cast<int64_t>(S_d_0) * S_d_1 * S_d_2;
  const int64_t total = static_cast<int64_t>(Bn) * H_all;

  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int     b    = static_cast<int>(idx / H_all);
  int64_t       rest = idx - static_cast<int64_t>(b) * H_all;
  const int     h0   = static_cast<int>(rest / H_12);
  rest -= static_cast<int64_t>(h0) * H_12;
  const int     h1 = static_cast<int>(rest / H_d_2);
  const int     h2 = static_cast<int>(rest - static_cast<int64_t>(h1) * H_d_2);

  const float* qs0 = qs_d_0 + (b * H_d_0 + h0) * S_d_0;
  const float* qs1 = qs_d_1 + (b * H_d_1 + h1) * S_d_1;
  const float* qs2 = qs_d_2 + (b * H_d_2 + h2) * S_d_2;

  const float* A_b   = A + static_cast<size_t>(b) * O * K_d;
  const float* wA_b  = USE_PA ? (wA + static_cast<size_t>(b) * O * K_d) : nullptr;
  const float* lin_b = USE_LINEAR ? (linear + static_cast<size_t>(b) * K_d) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;

  if (USE_STATES || USE_PA) {
    for (int o = 0; o < O; ++o) {
      const float* A_bo  = A_b + o * K_d;
      const float* wA_bo = USE_PA ? (wA_b + o * K_d) : nullptr;

      float qo_o = 0.0f;
      float wa_o = 0.0f;

      int k = 0;
      for (int s0 = 0; s0 < S_d_0; ++s0) {
        const float q0 = qs0[s0];
        for (int s1 = 0; s1 < S_d_1; ++s1) {
          const float q01 = q0 * qs1[s1];
          for (int s2 = 0; s2 < S_d_2; ++s2, ++k) {
            const float q = q01 * qs2[s2];
            qo_o += A_bo[k] * q;
            if (USE_PA) wa_o += wA_bo[k] * q;
          }
        }
      }

      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) pA_acc += qo_o * wa_o;
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    int k = 0;
    for (int s0 = 0; s0 < S_d_0; ++s0) {
      const float q0 = qs0[s0];
      for (int s1 = 0; s1 < S_d_1; ++s1) {
        const float q01 = q0 * qs1[s1];
        for (int s2 = 0; s2 < S_d_2; ++s2, ++k) {
          linear_acc += lin_b[k] * q01 * qs2[s2];
        }
      }
    }
  }

  const int64_t local_idx = (static_cast<int64_t>(h0) * H_d_1 + h1) * H_d_2 + h2;

  write_modality_score<USE_STATES>(score_out, b, b_stride, local_idx, acc_ent, linear_acc + pA_acc);
}

cudaError_t launch_modality_score_dedup_rank3_fused_tiny(const float* A_unflat, const float* wA_unflat,
                                                         const float* linear, const float* qs_d_0, const float* qs_d_1,
                                                         const float* qs_d_2, int Bn, int O, int H_d_0, int H_d_1,
                                                         int H_d_2, int S_d_0, int S_d_1, int S_d_2, int64_t b_stride,
                                                         bool use_states, bool use_linear, bool use_pA,
                                                         float* score_out, cudaStream_t stream) {
  PYMDP_LAUNCH_BOOL3_1D(static_cast<int64_t>(Bn) * H_d_0 * H_d_1 * H_d_2, use_states, use_linear, use_pA,
                        modality_score_dedup_rank3_fused_tiny_kernel,
                        A_unflat, wA_unflat, linear, qs_d_0, qs_d_1, qs_d_2, Bn, O, H_d_0, H_d_1, H_d_2, S_d_0, S_d_1,
                        S_d_2, b_stride, score_out);
}

// -----------------------------------------------------------------------------
// cuBLAS helper kernels (rank-2 / rank-3 modality). .cc launches them around
// cublasSgemmStridedBatched. Rank-3 stage 1 = build_qs01_outer → GEMM →
// tmp_qo_cublas_to_my → tmp_lin_per_h; rank-3 stage 2 (below) folds entropy
// + linear. Rank-2 reuses build_qs01_outer + GEMM, finishes via
// modality_score_dedup_rank2_cublas_finish_kernel below.
// -----------------------------------------------------------------------------

// q01[b, k_keep, h_kk] = qs_keep_0[b, h_0, s_0] * qs_keep_1[b, h_1, s_1].
// One thread per output element; layout matches the downstream GEMM B operand.
__global__ void build_qs01_outer_kernel(const float* __restrict__ qs_keep_0, const float* __restrict__ qs_keep_1,
                                        int Bn, int H_0, int H_1, int S_0, int S_1, float* __restrict__ q01_outer) {
  const int64_t idx    = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t K_keep = static_cast<int64_t>(S_0) * S_1;
  const int64_t H_kk   = static_cast<int64_t>(H_0) * H_1;
  const int64_t per_b  = K_keep * H_kk;
  const int64_t total  = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b   = static_cast<int>(idx / per_b);
  int64_t       r   = idx - static_cast<int64_t>(b) * per_b;
  const int     k   = static_cast<int>(r / H_kk);
  const int     h   = static_cast<int>(r - static_cast<int64_t>(k) * H_kk);
  const int     s_0 = k / S_1;
  const int     s_1 = k - s_0 * S_1;
  const int     h_0 = h / H_1;
  const int     h_1 = h - h_0 * H_1;

  const float qs0 = qs_keep_0[(static_cast<int64_t>(b) * H_0 + h_0) * S_0 + s_0];
  const float qs1 = qs_keep_1[(static_cast<int64_t>(b) * H_1 + h_1) * S_1 + s_1];
  q01_outer[idx]  = qs0 * qs1;
}

cudaError_t launch_build_qs01_outer(const float* qs_keep_0, const float* qs_keep_1, int Bn, int H_0, int H_1, int S_0,
                                    int S_1, float* q01_outer, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * S_0 * S_1 * H_0 * H_1,
                  build_qs01_outer_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                      qs_keep_0, qs_keep_1, Bn, H_0, H_1, S_0, S_1, q01_outer));
}

// Transpose tmp_qo_cublas[Bn, O*S_split, H_kk] → tmp_my[Bn, H_kk, O, S_split].
// One thread per output; stage-2 reads the latter for cache-friendly striding.
__global__ void tmp_qo_cublas_to_my_kernel(const float* __restrict__ tmp_cublas, int Bn, int O, int S_split, int H_kk,
                                           float* __restrict__ tmp_my) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_b = static_cast<int64_t>(H_kk) * O * S_split;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b   = static_cast<int>(idx / per_b);
  int64_t       r   = idx - static_cast<int64_t>(b) * per_b;
  const int64_t oS  = static_cast<int64_t>(O) * S_split;
  const int     h   = static_cast<int>(r / oS);
  r -= static_cast<int64_t>(h) * oS;
  const int     o   = static_cast<int>(r / S_split);
  const int     s   = static_cast<int>(r - static_cast<int64_t>(o) * S_split);

  // tmp_cublas[b, o*S_split + s, h] to tmp_my[b, h, o, s].
  const size_t cublas_off =
      static_cast<size_t>(b) * O * S_split * H_kk + static_cast<size_t>(o * S_split + s) * H_kk + h;
  tmp_my[idx]             = tmp_cublas[cublas_off];
}

cudaError_t launch_tmp_qo_cublas_to_my(const float* tmp_cublas, int Bn, int O, int S_split, int H_kk, float* tmp_my,
                                       cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * H_kk * O * S_split,
                  tmp_qo_cublas_to_my_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                      tmp_cublas, Bn, O, S_split, H_kk, tmp_my));
}

// tmp_lin[b, h, s2] = sum_k q01_outer[b, k, h] * linear[b, k*S_split + s2].
// One thread per output.
__global__ void tmp_lin_per_h_kernel(const float* __restrict__ q01_outer, const float* __restrict__ linear, int Bn,
                                     int K_keep, int H_kk, int S_split, float* __restrict__ tmp_lin) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_b = static_cast<int64_t>(H_kk) * S_split;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b  = static_cast<int>(idx / per_b);
  int64_t       r  = idx - static_cast<int64_t>(b) * per_b;
  const int     h  = static_cast<int>(r / S_split);
  const int     s2 = static_cast<int>(r - static_cast<int64_t>(h) * S_split);

  const float* q01_b    = q01_outer + static_cast<size_t>(b) * K_keep * H_kk;
  const float* linear_b = linear + static_cast<size_t>(b) * K_keep * S_split;
  float        acc      = 0.0f;
  for (int k = 0; k < K_keep; ++k) {
    acc += q01_b[k * H_kk + h] * linear_b[k * S_split + s2];
  }
  tmp_lin[idx] = acc;
}

cudaError_t launch_tmp_lin_per_h(const float* q01_outer, const float* linear, int Bn, int K_keep, int H_kk, int S_split,
                                 float* tmp_lin, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * H_kk * S_split,
                  tmp_lin_per_h_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                      q01_outer, linear, Bn, K_keep, H_kk, S_split, tmp_lin));
}

// Rank-2 cuBLAS finish. Reads tmp_qo[Bn, O, H_kk] from the cuBLAS GEMM,
// folds in entropy + linear (+ pA when USE_PA). One thread per (b, h_kk).
//
// O is a compile-time template parameter (mirroring rank-3 stage 2) so the
// o-loop fully unrolls; with O runtime, nvcc emits a serial-dependent load
// chain that bottlenecks on memory latency. Unrolling exposes O independent
// loads per thread for the entropy/pA pass — ILP fills the latency window.
template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_cublas_finish_kernel(const float* __restrict__ tmp_qo, const float* __restrict__ tmp_wa,
                                                const float* __restrict__ q01_outer, const float* __restrict__ linear,
                                                int Bn, int H_kk, int K_d, int64_t b_stride,
                                                float* __restrict__ score_out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * H_kk;
  if (idx >= total) return;
  const int b = static_cast<int>(idx / H_kk);
  const int h = static_cast<int>(idx % H_kk);

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
    const float* tmp_b = tmp_qo + static_cast<size_t>(b) * O_TPL * H_kk;
    const float* twa_b = USE_PA ? (tmp_wa + static_cast<size_t>(b) * O_TPL * H_kk) : nullptr;
#pragma unroll
    for (int o = 0; o < O_TPL; ++o) {
      const float qo = tmp_b[o * H_kk + h];
      if (USE_STATES) acc_ent += clamp_log(qo);
      if (USE_PA) pA_acc += qo * twa_b[o * H_kk + h];
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    const float* q01_b = q01_outer + static_cast<size_t>(b) * K_d * H_kk;
    const float* lin_b = linear + static_cast<size_t>(b) * K_d;
    // K_d is runtime (varies by modality) so we partial-unroll for ILP
    // without exploding the instruction cache footprint.
#pragma unroll 4
    for (int k = 0; k < K_d; ++k) linear_acc += q01_b[k * H_kk + h] * lin_b[k];
  }

  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

// Runtime-O fallback for modalities with O > 8 (the templated path's
// PYMDP_CUDA_DISPATCH_1_8 caps at 8 instantiations to keep binary size
// bounded). Without compile-time O nvcc can't fully unroll, but `#pragma
// unroll 4` still buys 4× ILP for latency hiding on the entropy/pA pass.
template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_cublas_finish_runtime_o_kernel(const float* __restrict__ tmp_qo,
                                                          const float* __restrict__ tmp_wa,
                                                          const float* __restrict__ q01_outer,
                                                          const float* __restrict__ linear, int Bn, int O, int H_kk,
                                                          int K_d, int64_t b_stride, float* __restrict__ score_out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * H_kk;
  if (idx >= total) return;
  const int b = static_cast<int>(idx / H_kk);
  const int h = static_cast<int>(idx % H_kk);

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
    const float* tmp_b = tmp_qo + static_cast<size_t>(b) * O * H_kk;
    const float* twa_b = USE_PA ? (tmp_wa + static_cast<size_t>(b) * O * H_kk) : nullptr;
#pragma unroll 4
    for (int o = 0; o < O; ++o) {
      const float qo = tmp_b[o * H_kk + h];
      if (USE_STATES) acc_ent += clamp_log(qo);
      if (USE_PA) pA_acc += qo * twa_b[o * H_kk + h];
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    const float* q01_b = q01_outer + static_cast<size_t>(b) * K_d * H_kk;
    const float* lin_b = linear + static_cast<size_t>(b) * K_d;
#pragma unroll 4
    for (int k = 0; k < K_d; ++k) linear_acc += q01_b[k * H_kk + h] * lin_b[k];
  }

  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

namespace {

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
inline void launch_rank2_cublas_finish_tpl(const float* tmp_qo, const float* tmp_wa, const float* q01_outer,
                                           const float* linear, int Bn, int H_kk, int K_d, int64_t b_stride,
                                           float* score_out, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(Bn) * H_kk;
  if (total <= 0) return;
  const int block_size = launch_block_size_1d(stream, total);
  const int blocks     = launch_blocks(total, block_size);
  modality_score_dedup_rank2_cublas_finish_kernel<O_TPL, USE_STATES, USE_LINEAR, USE_PA>
      <<<blocks, block_size, 0, stream>>>(tmp_qo, tmp_wa, q01_outer, linear, Bn, H_kk, K_d, b_stride, score_out);
}

}  // namespace

cudaError_t launch_modality_score_dedup_rank2_cublas_finish(const float* tmp_qo, const float* tmp_wa,
                                                            const float* q01_outer, const float* linear, int Bn, int O,
                                                            int H_kk, int K_d, int64_t b_stride, bool use_states,
                                                            bool use_linear, bool use_pA, float* score_out,
                                                            cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(Bn) * H_kk;
  if (total <= 0) return cudaSuccess;
  if (O <= 0) return cudaErrorInvalidValue;

  // O ∈ [1, 8]: fully unrolled templated path. O > 8: runtime fallback with
  // partial unroll — production modalities can have large O so we can't
  // reject here.
  if (O <= 8) {
#define RANK2_FINISH_BOOL_BODY(S, L, P)                                                                                \
  launch_rank2_cublas_finish_tpl<RANK2_FINISH_O_VAL, S, L, P>(tmp_qo, tmp_wa, q01_outer, linear, Bn, H_kk, K_d,        \
                                                              b_stride, score_out, stream)
#define RANK2_FINISH_CASE(N)                                                                                           \
  case N: {                                                                                                            \
    constexpr int RANK2_FINISH_O_VAL = N;                                                                              \
    PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, RANK2_FINISH_BOOL_BODY);                                      \
    break;                                                                                                             \
  }
    PYMDP_CUDA_DISPATCH_1_8(O, RANK2_FINISH_CASE)
#undef RANK2_FINISH_CASE
#undef RANK2_FINISH_BOOL_BODY
  } else {
    const int block_size = launch_block_size_1d(stream, total);
    const int blocks     = launch_blocks(total, block_size);
#define RANK2_FINISH_RT_BODY(S, L, P)                                                                                  \
  modality_score_dedup_rank2_cublas_finish_runtime_o_kernel<S, L, P>                                                   \
      <<<blocks, block_size, 0, stream>>>(tmp_qo, tmp_wa, q01_outer, linear, Bn, O, H_kk, K_d, b_stride, score_out)
    PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, RANK2_FINISH_RT_BODY);
#undef RANK2_FINISH_RT_BODY
  }
  return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Rank-3 stage 2 (finish + entropy + linear). One thread per (b, h_0, h_1,
// h_2). Per o: qo_o = sum_s tmp_qo[o, s] * qs_split[s]; fold into entropy.
// When USE_PA the same o-loop reduces wa_o = sum_s tmp_wa[o, s] * qs_split[s]
// and pA_acc += qo_o * wa_o. linear_acc = sum_s tmp_lin[s] * qs_split[s].
// In-modality flat index has h_keep_0 outermost, h_split innermost.
// -----------------------------------------------------------------------------

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank3_split_stage2_kernel(const float* __restrict__ tmp_qo, const float* __restrict__ tmp_lin,
                                               const float* __restrict__ tmp_wa, const float* __restrict__ qs_split,
                                               int Bn, int H_keep_0, int H_keep_1, int H_split, int S_split,
                                               int64_t b_stride, float* __restrict__ score_out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t H_12  = static_cast<int64_t>(H_keep_1) * H_split;
  const int64_t per_b = static_cast<int64_t>(H_keep_0) * H_12;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b    = static_cast<int>(idx / per_b);
  int64_t       rest = idx - static_cast<int64_t>(b) * per_b;
  const int     h_0  = static_cast<int>(rest / H_12);
  rest -= static_cast<int64_t>(h_0) * H_12;
  const int     h_1  = static_cast<int>(rest / H_split);
  const int     h_2  = static_cast<int>(rest - static_cast<int64_t>(h_1) * H_split);

  const int    total_h_keep = H_keep_0 * H_keep_1;
  const int    h_keep       = h_0 * H_keep_1 + h_1;
  const float* qs2_p        = qs_split + (b * H_split + h_2) * S_split;
  const float* tmp_qo_w     = (USE_STATES || USE_PA)
                                  ? (tmp_qo + ((static_cast<size_t>(b) * total_h_keep + h_keep) * O_TPL) * S_split)
                                  : nullptr;
  const float* tmp_wa_w =
      USE_PA ? (tmp_wa + ((static_cast<size_t>(b) * total_h_keep + h_keep) * O_TPL) * S_split) : nullptr;
  const float* tmp_lin_w =
      USE_LINEAR ? (tmp_lin + (static_cast<size_t>(b) * total_h_keep + h_keep) * S_split) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
#pragma unroll
    for (int o = 0; o < O_TPL; ++o) {
      float qo_o = 0.0f;
      for (int s = 0; s < S_split; ++s) qo_o += tmp_qo_w[o * S_split + s] * qs2_p[s];
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) {
        float wa_o = 0.0f;
        for (int s = 0; s < S_split; ++s) wa_o += tmp_wa_w[o * S_split + s] * qs2_p[s];
        pA_acc += qo_o * wa_o;
      }
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int s = 0; s < S_split; ++s) linear_acc += tmp_lin_w[s] * qs2_p[s];
  }

  // local_idx matches the pmi encoding in build_factor_history_tables.
  const int64_t local_idx = static_cast<int64_t>((h_0 * H_keep_1 + h_1)) * H_split + h_2;
  write_modality_score<USE_STATES>(score_out, b, b_stride, local_idx, acc_ent, linear_acc + pA_acc);
}

// Runtime-O fallback for rank-3 stage 2 (modalities with O > 8). Mirrors
// the rank-2 finish runtime-O fallback; partial-unrolled o-loop.
template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank3_split_stage2_runtime_o_kernel(const float* __restrict__ tmp_qo,
                                                         const float* __restrict__ tmp_lin,
                                                         const float* __restrict__ tmp_wa,
                                                         const float* __restrict__ qs_split, int Bn, int O,
                                                         int H_keep_0, int H_keep_1, int H_split, int S_split,
                                                         int64_t b_stride, float* __restrict__ score_out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t H_12  = static_cast<int64_t>(H_keep_1) * H_split;
  const int64_t per_b = static_cast<int64_t>(H_keep_0) * H_12;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int b    = static_cast<int>(idx / per_b);
  int64_t   rest = idx - static_cast<int64_t>(b) * per_b;
  const int h_0  = static_cast<int>(rest / H_12);
  rest -= static_cast<int64_t>(h_0) * H_12;
  const int h_1 = static_cast<int>(rest / H_split);
  const int h_2 = static_cast<int>(rest - static_cast<int64_t>(h_1) * H_split);

  const int    total_h_keep = H_keep_0 * H_keep_1;
  const int    h_keep       = h_0 * H_keep_1 + h_1;
  const float* qs2_p        = qs_split + (b * H_split + h_2) * S_split;
  const float* tmp_qo_w     = (USE_STATES || USE_PA)
                                  ? (tmp_qo + ((static_cast<size_t>(b) * total_h_keep + h_keep) * O) * S_split)
                                  : nullptr;
  const float* tmp_wa_w =
      USE_PA ? (tmp_wa + ((static_cast<size_t>(b) * total_h_keep + h_keep) * O) * S_split) : nullptr;
  const float* tmp_lin_w =
      USE_LINEAR ? (tmp_lin + (static_cast<size_t>(b) * total_h_keep + h_keep) * S_split) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
#pragma unroll 4
    for (int o = 0; o < O; ++o) {
      float qo_o = 0.0f;
      for (int s = 0; s < S_split; ++s) qo_o += tmp_qo_w[o * S_split + s] * qs2_p[s];
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) {
        float wa_o = 0.0f;
        for (int s = 0; s < S_split; ++s) wa_o += tmp_wa_w[o * S_split + s] * qs2_p[s];
        pA_acc += qo_o * wa_o;
      }
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int s = 0; s < S_split; ++s) linear_acc += tmp_lin_w[s] * qs2_p[s];
  }

  const int64_t local_idx = static_cast<int64_t>((h_0 * H_keep_1 + h_1)) * H_split + h_2;
  write_modality_score<USE_STATES>(score_out, b, b_stride, local_idx, acc_ent, linear_acc + pA_acc);
}

namespace {

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
inline void launch_rank3_stage2_tpl(const float* tmp_qo, const float* tmp_lin, const float* tmp_wa,
                                    const float* qs_split, int Bn, int H_keep_0, int H_keep_1, int H_split, int S_split,
                                    int64_t b_stride, float* score_out, cudaStream_t stream) {
  const int64_t total = static_cast<int64_t>(Bn) * H_keep_0 * H_keep_1 * H_split;
  if (total <= 0) return;
  // Wider blocks regardless of total: this is the finish kernel after a
  // cuBLAS stage-1 GEMM, and production shapes leave `total` in the low
  // hundreds — `launch_block_size_1d`'s 2048-element gate falls closed and
  // the grid lands at 1-5 blocks of 128 threads, badly under-occupying the
  // SMs. Going from 128 → 256 doubles the resident warps per block, which
  // is the only latency-hiding lever available when block count is fixed
  // by the small grid. The kernel is memory-latency-bound and has no shared
  // memory, so wider blocks have no occupancy downside. sm_87 caps at 256
  // to stay under the 16-block-per-SM limit on small grids; future arches
  // with larger register files could push to 512.
  constexpr int block_size = 256;
  const int     blocks     = launch_blocks(total, block_size);
  modality_score_dedup_rank3_split_stage2_kernel<O_TPL, USE_STATES, USE_LINEAR, USE_PA>
      <<<blocks, block_size, 0, stream>>>(tmp_qo, tmp_lin, tmp_wa, qs_split, Bn, H_keep_0, H_keep_1, H_split, S_split,
                                          b_stride, score_out);
}

}  // namespace

cudaError_t launch_modality_score_dedup_rank3_stage2(const float* tmp_qo, const float* tmp_lin, const float* tmp_wa,
                                                     const float* qs_split, int Bn, int O, int H_keep_0, int H_keep_1,
                                                     int H_split, int S_split, int64_t b_stride, bool use_states,
                                                     bool use_linear, bool use_pA, float* score_out,
                                                     cudaStream_t stream) {
  // Empty-grid guard in the public wrapper (same reason as
  // launch_b_rollout_general — see that wrapper).
  const int64_t total = static_cast<int64_t>(Bn) * H_keep_0 * H_keep_1 * H_split;
  if (total <= 0) return cudaSuccess;
  if (O <= 0) return cudaErrorInvalidValue;

  // O ∈ [1, 8]: fully unrolled templated path. O > 8: runtime fallback —
  // production modalities can have large O, so we can't reject here.
  if (O <= 8) {
#define RANK3_STAGE2_BOOL_BODY(S, L, P)                                                                                \
  launch_rank3_stage2_tpl<RANK3_STAGE2_O_VAL, S, L, P>(tmp_qo, tmp_lin, tmp_wa, qs_split, Bn, H_keep_0, H_keep_1,      \
                                                       H_split, S_split, b_stride, score_out, stream)
#define RANK3_STAGE2_CASE(N)                                                                                           \
  case N: {                                                                                                            \
    constexpr int RANK3_STAGE2_O_VAL = N;                                                                              \
    PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, RANK3_STAGE2_BOOL_BODY);                                      \
    break;                                                                                                             \
  }
    PYMDP_CUDA_DISPATCH_1_8(O, RANK3_STAGE2_CASE)
#undef RANK3_STAGE2_CASE
#undef RANK3_STAGE2_BOOL_BODY
  } else {
    constexpr int block_size = 256;  // Same wider block as the templated path.
    const int     blocks     = launch_blocks(total, block_size);
#define RANK3_STAGE2_RT_BODY(S, L, P)                                                                                  \
  modality_score_dedup_rank3_split_stage2_runtime_o_kernel<S, L, P><<<blocks, block_size, 0, stream>>>(                \
      tmp_qo, tmp_lin, tmp_wa, qs_split, Bn, O, H_keep_0, H_keep_1, H_split, S_split, b_stride, score_out)
    PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, RANK3_STAGE2_RT_BODY);
#undef RANK3_STAGE2_RT_BODY
  }
  return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Per-factor inductive coefficient v_full (mirrors precompute_inductive in
// neg_efe_precompute.h). One block per (b, f): thread 0 computes scalar
// reductions (argmax qs, best-depth m_f, path availability pa) into shmem;
// all threads then write per-s outputs in parallel.
// -----------------------------------------------------------------------------

__global__ void v_full_kernel(const float* __restrict__ qs_init, const float* __restrict__ I,
                              const float* __restrict__ eps, int eps_stride, int Bn, int F, int qs_flat,
                              int64_t I_per_batch, const int32_t* __restrict__ S, const int32_t* __restrict__ depth,
                              const int32_t* __restrict__ qs_off, const int32_t* __restrict__ I_off,
                              float* __restrict__ v_out) {
  const int b = blockIdx.x;
  const int f = blockIdx.y;
  if (b >= Bn || f >= F) return;

  const int Sf       = S[f];
  const int Df       = depth[f];
  const int qs_off_f = qs_off[f];
  const int I_off_f  = I_off[f];

  const float* qsf = qs_init + static_cast<int64_t>(b) * qs_flat + qs_off_f;
  const float* If  = I + static_cast<int64_t>(b) * I_per_batch + I_off_f;

  // Single-warp block (launcher uses 32 threads). All three scalar
  // reductions (qs argmax, I-depth argmax, pa sum) run cooperatively
  // across the warp's 32 lanes.
  __shared__ int   s_mf;
  __shared__ float s_path_log_eps;

  // Step 1: argmax over qsf[0..Sf]. Each lane handles ⌈Sf/32⌉ slots;
  // tie-break on smallest idx so the result is deterministic regardless
  // of which lane saw the max first.
  float val = -INFINITY;
  int   idx = 0;
  for (int s = threadIdx.x; s < Sf; s += 32) {
    const float v = qsf[s];
    if (v > val || (v == val && s < idx)) {
      val = v;
      idx = s;
    }
  }
  warp_reduce_argmax(val, idx);
  // All lanes now hold the same `idx` (the qs argmax).

  // Step 2: argmax over If[0..Df, idx]. Df is small (typically ≤ policy_len),
  // so most lanes contribute -inf; the reduce still converges in 5 shuffles.
  float mval = -INFINITY;
  int   midx = 0;
  for (int i = threadIdx.x; i < Df; i += 32) {
    const float v = If[i * Sf + idx];
    if (v > mval || (v == mval && i < midx)) {
      mval = v;
      midx = i;
    }
  }
  warp_reduce_argmax(mval, midx);

  // Step 3: pa = sum_i If[i, idx]. Same shuffle pattern.
  float pa = 0.0f;
  for (int i = threadIdx.x; i < Df; i += 32) {
    pa += If[i * Sf + idx];
  }
  pa = warp0_shfl_sum(pa);  // XOR-butterfly: result lands on every lane

  // Lane 0 finalizes scalar dependent values and publishes via shmem so
  // the broadcast read below sees them on all lanes regardless of arch.
  if (threadIdx.x == 0) {
    int m_f = midx;
    if (m_f > 0) m_f -= 1;
    s_mf = m_f;

    const float pa_clamped = fmaxf(0.0f, fminf(1.0f, pa));
    // NaN-trap guard. logf(0) = -inf and 0 * -inf = NaN, which would
    // contaminate v_p and bleed through into the final scatter. The host
    // path validates eps in check_inductive_epsilon_value, but the dev
    // path skips that to avoid a D2H sync — so this kernel is the last
    // line of defense. The pa <= 0 short-circuit also skips logf when the
    // factor has no inductive path.
    if (pa_clamped <= 0.0f) {
      s_path_log_eps = 0.0f;
    } else {
      const float eps_val = fmaxf(eps[b * eps_stride], 1e-16f);
      s_path_log_eps      = pa_clamped * logf(eps_val);
    }
  }
  __syncthreads();

  const int    mf    = s_mf;
  const float  scale = s_path_log_eps;
  float* const v_p   = v_out + static_cast<int64_t>(b) * qs_flat + qs_off_f;
  const float* I_mf  = If + mf * Sf;
  for (int s = threadIdx.x; s < Sf; s += blockDim.x) {
    v_p[s] = scale * (1.0f - I_mf[s]);
  }
}

cudaError_t launch_v_full(const float* qs_init, const float* I, const float* eps, int eps_stride, int Bn, int F,
                          int qs_flat, int64_t I_per_batch, const int32_t* S, const int32_t* depth,
                          const int32_t* qs_off, const int32_t* I_off, float* v_out, cudaStream_t stream) {
  if (Bn <= 0 || F <= 0) return cudaSuccess;
  // Block size 32 covers typical S_f; larger Sf falls back to strided per-thread.
  v_full_kernel<<<dim3(Bn, F), 32, 0, stream>>>(qs_init, I, eps, eps_stride, Bn, F, qs_flat, I_per_batch, S, depth,
                                                qs_off, I_off, v_out);
  return cudaGetLastError();
}

// -----------------------------------------------------------------------------
// Final scatter (factor-history form). One thread per (b, p):
//   out[b, p] = sum_(t,m) scores_concat[b*total_mod + mod_off[t*M+m]
//                                                  + pmi[t*M*P + m*P + p]]
//             + sum_(t,f) inductive_concat[b*total_ind + ind_off[t*F+f]
//                                                  + p2h[t*F*P + f*P + p]]   (if use_inductive)
//             + sum_(t,f) factor_scores  [b*total_ind + ind_off[t*F+f]
//                                                  + p2h[t*F*P + f*P + p]]   (if use_factor_scores)
// inductive_concat and factor_scores share ind_off / p2h (same per-(t, f)
// layout).
// -----------------------------------------------------------------------------

template <bool USE_INDUCTIVE, bool USE_FACTOR_SCORES>
__global__ void
final_scatter_dedup_kernel(const float* __restrict__ scores_concat, const float* __restrict__ inductive_concat,
                           const float* __restrict__ factor_scores, const int32_t* __restrict__ policy_to_modality_idx,
                           const int32_t* __restrict__ policy_to_factor_history, const int64_t* __restrict__ mod_off,
                           const int64_t* __restrict__ ind_off, int Bn, int T, int M, int F, int P, int64_t total_mod,
                           int64_t total_ind, float* __restrict__ out) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * P;
  if (idx >= total) return;
  const int b = static_cast<int>(idx / P);
  const int p = static_cast<int>(idx % P);

  // Per-batch bases in int64; the t/m/f walks then index policy_to_* in int32
  // (production T*M*P / T*F*P stays well under 2^31).
  const float* __restrict__ scores_b        = scores_concat + b * total_mod;
  const float* __restrict__ inductive_b     = USE_INDUCTIVE ? (inductive_concat + b * total_ind) : nullptr;
  const float* __restrict__ factor_scores_b = USE_FACTOR_SCORES ? (factor_scores + b * total_ind) : nullptr;

  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    const int t_M   = t * M;
    const int t_M_P = t_M * P;
    for (int m = 0; m < M; ++m) {
      const int64_t off      = mod_off[t_M + m];
      const int32_t flat_idx = policy_to_modality_idx[t_M_P + m * P + p];
      acc += scores_b[off + flat_idx];
    }
    if (USE_INDUCTIVE || USE_FACTOR_SCORES) {
      const int t_F   = t * F;
      const int t_F_P = t_F * P;
      if (USE_INDUCTIVE) {
        for (int f = 0; f < F; ++f) {
          const int64_t off = ind_off[t_F + f];
          const int32_t h_f = policy_to_factor_history[t_F_P + f * P + p];
          acc += inductive_b[off + h_f];
        }
      }
      if (USE_FACTOR_SCORES) {
        for (int f = 0; f < F; ++f) {
          const int64_t off = ind_off[t_F + f];
          const int32_t h_f = policy_to_factor_history[t_F_P + f * P + p];
          acc += factor_scores_b[off + h_f];
        }
      }
    }
  }
  out[idx] = acc;
}

cudaError_t launch_final_scatter_dedup(const float* scores_concat, const float* inductive_concat,
                                       const float* factor_scores, const int32_t* policy_to_modality_idx,
                                       const int32_t* policy_to_factor_history, const int64_t* mod_off,
                                       const int64_t* ind_off, int Bn, int T, int M, int F, int P, int64_t total_mod,
                                       int64_t total_ind, bool use_inductive, bool use_factor_scores, float* out,
                                       cudaStream_t stream) {
  PYMDP_LAUNCH_BOOL2_1D(static_cast<int64_t>(Bn) * P, use_inductive, use_factor_scores, final_scatter_dedup_kernel,
                        scores_concat, inductive_concat, factor_scores, policy_to_modality_idx,
                        policy_to_factor_history, mod_off, ind_off, Bn, T, M, F, P, total_mod, total_ind, out);
}

}  // namespace cuda_kernels
}  // namespace pymdp_ffi
