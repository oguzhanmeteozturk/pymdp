// Device kernels for CUDA neg-EFE: B-rollout, modality scoring (rank 1/2/3),
// inductive v_full, final scatter, and content-tag gather. Compiled by nvcc.
// Supported: Maxwell sm_53–Ampere sm_87; warp-shuffle gates on __CUDA_ARCH__.

#include "neg_efe/neg_efe_cuda_kernels.h"

#include <cstdint>

#include "common/cuda_warp_reduce.h"

namespace pymdp_ffi {
namespace cuda_kernels {

// -----------------------------------------------------------------------------
// Common helpers
// -----------------------------------------------------------------------------

// Block-size tuning: sm_87 gets 256 for 1-D scoring (latency hiding); 128 elsewhere.
// B-rollout stays at 128 (shmem + reduction schedules tuned to it).
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

// Cached compute-capability lookup (thread_local); returns 0 on error.
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

// Use 256 threads on sm_87+ only when total >= 2048 (else waste; scheduler ping faster on 128).
constexpr int64_t kAmpereWideMinTotal = 2048;

inline int launch_block_size_1d(cudaStream_t /*stream*/, int64_t total) {
  if (device_cc_cached() < 87) return kBlockSizeDefault;
  return (total >= kAmpereWideMinTotal) ? kBlockSizeAmpere1D : kBlockSizeDefault;
}

// 1-D grid launcher macro. Guard empty grid, compute blocks, call kernel with pymdp_block_size.
// Must be terminal statement in a cudaError_t-returning function; returns cudaGetLastError().
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
    const int pymdp_b3 = (static_cast<int>(use_s) << 2) | (static_cast<int>(use_l) << 1) | static_cast<int>(use_p);    \
    switch (pymdp_b3) {                                                                                                \
    case 0:                                                                                                            \
      kernel<false, false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                         \
      break;                                                                                                           \
    case 1:                                                                                                            \
      kernel<false, false, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                          \
      break;                                                                                                           \
    case 2:                                                                                                            \
      kernel<false, true, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                          \
      break;                                                                                                           \
    case 3:                                                                                                            \
      kernel<false, true, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                           \
      break;                                                                                                           \
    case 4:                                                                                                            \
      kernel<true, false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                          \
      break;                                                                                                           \
    case 5:                                                                                                            \
      kernel<true, false, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                           \
      break;                                                                                                           \
    case 6:                                                                                                            \
      kernel<true, true, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                           \
      break;                                                                                                           \
    case 7:                                                                                                            \
      kernel<true, true, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                            \
      break;                                                                                                           \
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
    case 0:                                                                                                            \
      kernel<false, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                                \
      break;                                                                                                           \
    case 1:                                                                                                            \
      kernel<false, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                                 \
      break;                                                                                                           \
    case 2:                                                                                                            \
      kernel<true, false><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                                 \
      break;                                                                                                           \
    case 3:                                                                                                            \
      kernel<true, true><<<pymdp_blocks, pymdp_block_size, 0, stream>>>(__VA_ARGS__);                                  \
      break;                                                                                                           \
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
// Shmem: dynamic, K_f * 4B per block. configure_b_rollout_for_arch raises the
// per-launch ceiling to the arch max (48 KB on sm_53 -> K_f <= 12288; up to
// ~164 KB on sm_87); K_f beyond the arch max trips cudaErrorInvalidConfiguration.
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

// Warp-wide argmax: each lane brings (val, idx); all lanes converge on maximum.
// Ties broken by smaller idx (deterministic across runs).
__device__ __forceinline__ void warp_reduce_argmax(float& val, int& idx) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    const float other_val = __shfl_xor_sync(0xffffffffu, val, off);
    const int   other_idx = __shfl_xor_sync(0xffffffffu, idx, off);
    if (other_val > val || (other_val == val && other_idx < idx)) {
      val = other_val;
      idx = other_idx;
    }
  }
}

// block_reduce_sum_lane0, block_reduce_sum_pair_lane0 hoisted to common/cuda_warp_reduce.h

template <int BLOCK_SIZE, bool COMPUTE_WB, bool COMPUTE_INDUCTIVE>
__device__ __forceinline__ void b_rollout_phase2_and_wb(const float* __restrict__ B, const float* __restrict__ wB_tr,
                                                        const float* __restrict__ v_full_b, int b, int h, int u, int Nh,
                                                        int S_f, int K_f, int U_f, const float* __restrict__ qs_outer,
                                                        float* __restrict__ qs_out, float* __restrict__ factor_score,
                                                        float* __restrict__ ind_score_t_f, int64_t ind_b_stride) {
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
__global__ void
b_rollout_general_kernel(const float* __restrict__ B, const float* __restrict__ wB_tr, const float* __restrict__ v_full,
                         int qs_flat, int qs_off_f, const int32_t* __restrict__ action_h,
                         const int32_t* __restrict__ parent_histories, BRolloutParents parents, int Bn, int Nh, int S_f,
                         int K_f, int U_f, float* __restrict__ qs_out, float* __restrict__ factor_score,
                         float* __restrict__ ind_score_t_f, int64_t ind_b_stride) {
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
    const int S1                     = parents.S[1];
    const float* __restrict__ qs0_bh = parent_qs_base(parents, 0, b, parent_histories[h * 2 + 0]);
    const float* __restrict__ qs1_bh = parent_qs_base(parents, 1, b, parent_histories[h * 2 + 1]);
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      const int s_0 = k / S1;
      const int s_1 = k - s_0 * S1;
      qs_outer[k]   = qs0_bh[s_0] * qs1_bh[s_1];
    }
  } else if (N_PARENTS == 3) {
    const int S1                     = parents.S[1];
    const int S2                     = parents.S[2];
    const int S_12                   = S1 * S2;
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

  const float* v_full_b = COMPUTE_INDUCTIVE ? (v_full + static_cast<size_t>(b) * qs_flat + qs_off_f) : nullptr;
  b_rollout_phase2_and_wb<BLOCK_SIZE, COMPUTE_WB, COMPUTE_INDUCTIVE>(
      B, wB_tr, v_full_b, b, h, u, Nh, S_f, K_f, U_f, qs_outer, qs_out, factor_score, ind_score_t_f, ind_b_stride);
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
  auto               kernel = b_rollout_general_kernel<BLOCK_SIZE, N_PARENTS, COMPUTE_WB, COMPUTE_INDUCTIVE>;
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
                                        float* qs_out, float* factor_score, float* ind_score_t_f, int64_t ind_b_stride,
                                        cudaStream_t stream) {
  // dim3(0, ...) trips cudaErrorInvalidConfiguration on some drivers.
  if (Bn <= 0 || Nh <= 0) return cudaSuccess;
  const dim3   grid(static_cast<unsigned>(Bn), static_cast<unsigned>(Nh), 1);
  const size_t shmem_bytes = static_cast<size_t>(K_f) * sizeof(float);
  cudaError_t  err =
      configure_b_rollout_for_arch<kBRolloutBlockSize, N_PARENTS, COMPUTE_WB, COMPUTE_INDUCTIVE>(shmem_bytes);
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
  pymdp_br_rc = launch_b_rollout_tpl<PYMDP_BR_N_VAL, W, I>(B, wB_tr, v_full, qs_flat, qs_off_f, action_h,              \
                                                           parent_histories, parents, Bn, Nh, S_f, K_f, U_f, qs_out,   \
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
                        modality_score_dedup_rank1_kernel, A_unflat, wA_unflat, linear, qs_d_0, Bn, O, H_d_0, S_d_0,
                        b_stride, score_out);
}

// -----------------------------------------------------------------------------
// Tiny-shape fused rank-2 / rank-3 scoring. Inline s-Kronecker per (b, h);
// no q01_outer / no cuBLAS / no tmp buffers. Gated by the host heuristic
// `use_tiny_fused_modality_path` (in neg_efe_cuda_launch.cc); larger shapes fall
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
                        modality_score_dedup_rank2_fused_tiny_kernel, A_unflat, wA_unflat, linear, qs_d_0, qs_d_1, Bn,
                        O, H_d_0, H_d_1, S_d_0, S_d_1, b_stride, score_out);
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

  const int b    = static_cast<int>(idx / H_all);
  int64_t   rest = idx - static_cast<int64_t>(b) * H_all;
  const int h0   = static_cast<int>(rest / H_12);
  rest -= static_cast<int64_t>(h0) * H_12;
  const int h1 = static_cast<int>(rest / H_d_2);
  const int h2 = static_cast<int>(rest - static_cast<int64_t>(h1) * H_d_2);

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
                        modality_score_dedup_rank3_fused_tiny_kernel, A_unflat, wA_unflat, linear, qs_d_0, qs_d_1,
                        qs_d_2, Bn, O, H_d_0, H_d_1, H_d_2, S_d_0, S_d_1, S_d_2, b_stride, score_out);
}

// -----------------------------------------------------------------------------
// cuBLAS helper kernels (rank-2 / rank-3 modality). .cc launches them around
// cublasSgemmStridedBatched. Rank-3 stage 1 = build_qs01_outer → GEMM →
// tmp_qo_cublas_to_my → tmp_lin_per_h; rank-3 stage 2 (below) folds entropy
// + linear. Rank-2 reuses build_qs01_outer + an A GEMM (+ a second GEMM for
// the linear term, linear @ q01_outer → tmp_lin), finishing via
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

  const int b   = static_cast<int>(idx / per_b);
  int64_t   r   = idx - static_cast<int64_t>(b) * per_b;
  const int k   = static_cast<int>(r / H_kk);
  const int h   = static_cast<int>(r - static_cast<int64_t>(k) * H_kk);
  const int s_0 = k / S_1;
  const int s_1 = k - s_0 * S_1;
  const int h_0 = h / H_1;
  const int h_1 = h - h_0 * H_1;

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

// Small-shape batched GEMM (row-major): out[b, m, n] = sum_k A[b, m, k] * Q[b, k, n].
// One warp per output element; the 32 lanes stride-reduce over K. Built for the
// small-M*N / large-K regime (rank-3 stage-1 with tiny H_kk), where cuBLAS picks
// a 128x128-tiled sgemm that leaves nearly the whole N dimension idle — ~100 us
// on shapes whose real work is a few microseconds. The warp-per-output reduction
// keeps full occupancy even when M*N is small but K is large.
__global__ void small_batched_gemm_rm_kernel(const float* __restrict__ A, const float* __restrict__ Q,
                                             float* __restrict__ out, int Bn, int M, int N, int K) {
  const int     warp  = static_cast<int>((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int     lane  = threadIdx.x & 31;
  const int64_t MN    = static_cast<int64_t>(M) * N;
  const int64_t total = static_cast<int64_t>(Bn) * MN;
  if (warp >= total) return;  // whole warp shares `warp`, so warp_reduce_sum below is never partial

  const int     b = static_cast<int>(warp / MN);
  const int64_t r = warp - static_cast<int64_t>(b) * MN;
  const int     m = static_cast<int>(r / N);
  const int     n = static_cast<int>(r - static_cast<int64_t>(m) * N);

  const float* __restrict__ A_bm = A + (static_cast<size_t>(b) * M + m) * K;  // A[b, m, :], contiguous in k
  const float* __restrict__ Q_bn = Q + static_cast<size_t>(b) * K * N + n;     // Q[b, :, n], stride N over k
  float acc = 0.0f;
  for (int k = lane; k < K; k += 32) acc += A_bm[k] * Q_bn[static_cast<size_t>(k) * N];
  acc = warp_reduce_sum(acc);
  if (lane == 0) out[static_cast<size_t>(b) * MN + r] = acc;
}

cudaError_t launch_small_batched_gemm_rm(const float* a_rm, const float* q01_outer, float* out_rm, int Bn, int M, int N,
                                         int K, cudaStream_t stream) {
  const int64_t total_warps = static_cast<int64_t>(Bn) * M * N;
  if (total_warps <= 0 || K <= 0) return cudaSuccess;
  constexpr int block         = 128;  // 4 warps/block
  const int64_t total_threads = total_warps * 32;
  const int     blocks        = static_cast<int>((total_threads + block - 1) / block);
  small_batched_gemm_rm_kernel<<<blocks, block, 0, stream>>>(a_rm, q01_outer, out_rm, Bn, M, N, K);
  return cudaGetLastError();
}

// Transpose tmp_qo_cublas[Bn, O*S_split, H_kk] → tmp_my[Bn, H_kk, O, S_split].
// One thread per output; stage-2 reads the latter for cache-friendly striding.
__global__ void tmp_qo_cublas_to_my_kernel(const float* __restrict__ tmp_cublas, int Bn, int O, int S_split, int H_kk,
                                           float* __restrict__ tmp_my) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_b = static_cast<int64_t>(H_kk) * O * S_split;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b  = static_cast<int>(idx / per_b);
  int64_t       r  = idx - static_cast<int64_t>(b) * per_b;
  const int64_t oS = static_cast<int64_t>(O) * S_split;
  const int     h  = static_cast<int>(r / oS);
  r -= static_cast<int64_t>(h) * oS;
  const int o = static_cast<int>(r / S_split);
  const int s = static_cast<int>(r - static_cast<int64_t>(o) * S_split);

  // tmp_cublas[b, o*S_split + s, h] to tmp_my[b, h, o, s].
  const size_t cublas_off =
      static_cast<size_t>(b) * O * S_split * H_kk + static_cast<size_t>(o * S_split + s) * H_kk + h;
  tmp_my[idx] = tmp_cublas[cublas_off];
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

  const int b  = static_cast<int>(idx / per_b);
  int64_t   r  = idx - static_cast<int64_t>(b) * per_b;
  const int h  = static_cast<int>(r / S_split);
  const int s2 = static_cast<int>(r - static_cast<int64_t>(h) * S_split);

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

// Same contraction as tmp_lin_per_h_kernel, but one warp per output (b, h, s2)
// with the 32 lanes stride-reducing over K_keep. tmp_lin_per_h gives one thread
// per output and a serial K loop; when H_kk*S_split is small (the short
// policy_len regime) that under-occupies the GPU and the K_keep loop dominates
// (~155 us in the rollout_loop profile). Warp-parallel K keeps full occupancy.
__global__ void tmp_lin_warp_kernel(const float* __restrict__ q01_outer, const float* __restrict__ linear, int Bn,
                                    int K_keep, int H_kk, int S_split, float* __restrict__ tmp_lin) {
  const int     warp  = static_cast<int>((static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5);
  const int     lane  = threadIdx.x & 31;
  const int64_t HS    = static_cast<int64_t>(H_kk) * S_split;
  const int64_t total = static_cast<int64_t>(Bn) * HS;
  if (warp >= total) return;  // whole warp shares `warp`; warp_reduce_sum below is never partial

  const int     b  = static_cast<int>(warp / HS);
  const int64_t r  = warp - static_cast<int64_t>(b) * HS;
  const int     h  = static_cast<int>(r / S_split);
  const int     s2 = static_cast<int>(r - static_cast<int64_t>(h) * S_split);

  const float* __restrict__ q01_b = q01_outer + static_cast<size_t>(b) * K_keep * H_kk;
  const float* __restrict__ lin_b = linear + static_cast<size_t>(b) * K_keep * S_split;
  float acc = 0.0f;
  for (int k = lane; k < K_keep; k += 32) {
    acc += q01_b[static_cast<size_t>(k) * H_kk + h] * lin_b[static_cast<size_t>(k) * S_split + s2];
  }
  acc = warp_reduce_sum(acc);
  if (lane == 0) tmp_lin[static_cast<size_t>(b) * HS + r] = acc;
}

cudaError_t launch_tmp_lin_warp(const float* q01_outer, const float* linear, int Bn, int K_keep, int H_kk, int S_split,
                                float* tmp_lin, cudaStream_t stream) {
  const int64_t total_warps = static_cast<int64_t>(Bn) * H_kk * S_split;
  if (total_warps <= 0 || K_keep <= 0) return cudaSuccess;
  constexpr int block         = 128;  // 4 warps/block
  const int64_t total_threads = total_warps * 32;
  const int     blocks        = static_cast<int>((total_threads + block - 1) / block);
  tmp_lin_warp_kernel<<<blocks, block, 0, stream>>>(q01_outer, linear, Bn, K_keep, H_kk, S_split, tmp_lin);
  return cudaGetLastError();
}

// Rank-2 cuBLAS finish. Reads tmp_qo[Bn, O, H_kk] from the cuBLAS GEMM,
// folds in entropy + linear (+ pA when USE_PA). One thread per (b, h_kk).
//
// The linear term sum_k q01[b,k,h]*linear[b,k] is precomputed into
// tmp_lin[Bn, H_kk] by a cuBLAS GEMM (linear @ q01_outer) in the launcher —
// the K_d reduction must parallelize across threads, and one-thread-per-(b,h)
// here only exposes Bn*H_kk threads (64 in the high-dim/few-policy regime),
// which serially looping K_d=O(4096) leaves the GPU ~97% idle.
//
template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_cublas_finish_kernel(const float* __restrict__ tmp_qo, const float* __restrict__ tmp_wa,
                                                const float* __restrict__ tmp_lin, int Bn, int H_kk, int lin_b_stride,
                                                int64_t b_stride, float* __restrict__ score_out) {
  const int b = blockIdx.y;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H_kk) return;

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

  float linear_acc = USE_LINEAR ? tmp_lin[static_cast<size_t>(b) * lin_b_stride + h] : 0.0f;
  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank2_cublas_finish_runtime_o_kernel(
    const float* __restrict__ tmp_qo, const float* __restrict__ tmp_wa, const float* __restrict__ tmp_lin, int Bn,
    int O, int H_kk, int lin_b_stride, int64_t b_stride, float* __restrict__ score_out) {
  const int b = blockIdx.y;
  const int h = blockIdx.x * blockDim.x + threadIdx.x;
  if (h >= H_kk) return;

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

  float linear_acc = USE_LINEAR ? tmp_lin[static_cast<size_t>(b) * lin_b_stride + h] : 0.0f;
  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

namespace {

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
inline void launch_rank2_cublas_finish_tpl(const float* tmp_qo, const float* tmp_wa, const float* tmp_lin, int Bn,
                                           int H_kk, int lin_b_stride, int64_t b_stride, float* score_out,
                                           cudaStream_t stream) {
  if (Bn <= 0 || H_kk <= 0) return;
  const int64_t total      = static_cast<int64_t>(Bn) * H_kk;
  const int     block_size = launch_block_size_1d(stream, total);
  // 2-D grid keeps b on blockIdx.y so the kernel skips the 64-bit (b, h) decode.
  const dim3 grid(static_cast<unsigned>(launch_blocks(H_kk, block_size)), static_cast<unsigned>(Bn));
  modality_score_dedup_rank2_cublas_finish_kernel<O_TPL, USE_STATES, USE_LINEAR, USE_PA>
      <<<grid, block_size, 0, stream>>>(tmp_qo, tmp_wa, tmp_lin, Bn, H_kk, lin_b_stride, b_stride, score_out);
}

}  // namespace

cudaError_t launch_modality_score_dedup_rank2_cublas_finish(const float* tmp_qo, const float* tmp_wa,
                                                            const float* tmp_lin, int Bn, int O, int H_kk,
                                                            int lin_b_stride, int64_t b_stride, bool use_states,
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
  launch_rank2_cublas_finish_tpl<RANK2_FINISH_O_VAL, S, L, P>(tmp_qo, tmp_wa, tmp_lin, Bn, H_kk, lin_b_stride,         \
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
    const int  block_size = launch_block_size_1d(stream, total);
    const dim3 grid(static_cast<unsigned>(launch_blocks(H_kk, block_size)), static_cast<unsigned>(Bn));
#define RANK2_FINISH_RT_BODY(S, L, P)                                                                                  \
  modality_score_dedup_rank2_cublas_finish_runtime_o_kernel<S, L, P>                                                   \
      <<<grid, block_size, 0, stream>>>(tmp_qo, tmp_wa, tmp_lin, Bn, O, H_kk, lin_b_stride, b_stride, score_out)
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
  const int b     = blockIdx.y;
  const int inner = blockIdx.x * blockDim.x + threadIdx.x;
  const int H_12  = H_keep_1 * H_split;
  const int per_b = H_keep_0 * H_12;
  if (inner >= per_b) return;

  const int h_0  = inner / H_12;
  int       rest = inner - h_0 * H_12;
  const int h_1  = rest / H_split;
  const int h_2  = rest - h_1 * H_split;

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
      float wa_o = 0.0f;
      for (int s = 0; s < S_split; ++s) {
        const float q = qs2_p[s];
        qo_o += tmp_qo_w[o * S_split + s] * q;
        if (USE_PA) wa_o += tmp_wa_w[o * S_split + s] * q;
      }
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) pA_acc += qo_o * wa_o;
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int s = 0; s < S_split; ++s) linear_acc += tmp_lin_w[s] * qs2_p[s];
  }

  const int64_t local_idx = static_cast<int64_t>((h_0 * H_keep_1 + h_1)) * H_split + h_2;
  write_modality_score<USE_STATES>(score_out, b, b_stride, local_idx, acc_ent, linear_acc + pA_acc);
}

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank3_split_stage2_runtime_o_kernel(
    const float* __restrict__ tmp_qo, const float* __restrict__ tmp_lin, const float* __restrict__ tmp_wa,
    const float* __restrict__ qs_split, int Bn, int O, int H_keep_0, int H_keep_1, int H_split, int S_split,
    int64_t b_stride, float* __restrict__ score_out) {
  const int b     = blockIdx.y;
  const int inner = blockIdx.x * blockDim.x + threadIdx.x;
  const int H_12  = H_keep_1 * H_split;
  const int per_b = H_keep_0 * H_12;
  if (inner >= per_b) return;

  const int h_0  = inner / H_12;
  int       rest = inner - h_0 * H_12;
  const int h_1  = rest / H_split;
  const int h_2  = rest - h_1 * H_split;

  const int    total_h_keep = H_keep_0 * H_keep_1;
  const int    h_keep       = h_0 * H_keep_1 + h_1;
  const float* qs2_p        = qs_split + (b * H_split + h_2) * S_split;
  const float* tmp_qo_w =
      (USE_STATES || USE_PA) ? (tmp_qo + ((static_cast<size_t>(b) * total_h_keep + h_keep) * O) * S_split) : nullptr;
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
      float wa_o = 0.0f;
      for (int s = 0; s < S_split; ++s) {
        const float q = qs2_p[s];
        qo_o += tmp_qo_w[o * S_split + s] * q;
        if (USE_PA) wa_o += tmp_wa_w[o * S_split + s] * q;
      }
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) pA_acc += qo_o * wa_o;
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
  if (Bn <= 0 || H_keep_0 <= 0 || H_keep_1 <= 0 || H_split <= 0) return;
  // Wider blocks regardless of total: this is the finish kernel after a
  // cuBLAS stage-1 GEMM, and production shapes leave per-batch work in the
  // low hundreds — `launch_block_size_1d`'s 2048-element gate falls closed
  // and the grid lands at 1-5 blocks of 128 threads, badly under-occupying
  // the SMs. Going from 128 → 256 doubles the resident warps per block,
  // which is the only latency-hiding lever available when block count is
  // fixed by the small grid. The kernel is memory-latency-bound and has no
  // shared memory, so wider blocks have no occupancy downside. sm_87 caps
  // at 256 to stay under the 16-block-per-SM limit on small grids; future
  // arches with larger register files could push to 512.
  constexpr int block_size = 256;
  const int     per_b      = H_keep_0 * H_keep_1 * H_split;
  // 2-D grid keeps b on blockIdx.y so the kernel skips the 64-bit decode.
  const dim3 grid(static_cast<unsigned>(launch_blocks(per_b, block_size)), static_cast<unsigned>(Bn));
  modality_score_dedup_rank3_split_stage2_kernel<O_TPL, USE_STATES, USE_LINEAR, USE_PA>
      <<<grid, block_size, 0, stream>>>(tmp_qo, tmp_lin, tmp_wa, qs_split, Bn, H_keep_0, H_keep_1, H_split, S_split,
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
    const int     per_b      = H_keep_0 * H_keep_1 * H_split;
    const dim3    grid(static_cast<unsigned>(launch_blocks(per_b, block_size)), static_cast<unsigned>(Bn));
#define RANK3_STAGE2_RT_BODY(S, L, P)                                                                                  \
  modality_score_dedup_rank3_split_stage2_runtime_o_kernel<S, L, P><<<grid, block_size, 0, stream>>>(                  \
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

  // Step 2+3 fused: each lane walks If[i, idx] once, contributing to both
  // the depth argmax and the pa sum. Df is small (typically ≤ policy_len)
  // so most lanes contribute -inf / 0; both reduces still converge in 5
  // shuffles after the loop.
  float mval = -INFINITY;
  int   midx = 0;
  float pa   = 0.0f;
  for (int i = threadIdx.x; i < Df; i += 32) {
    const float v = If[i * Sf + idx];
    pa += v;
    if (v > mval || (v == mval && i < midx)) {
      mval = v;
      midx = i;
    }
  }
  warp_reduce_argmax(mval, midx);
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
  // Block is a single warp (32-thread launch), so __syncwarp is sufficient
  // for the shmem broadcast — no full block barrier needed.
  __syncwarp();

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
  // 2-D grid: blockIdx.y = b, blockIdx.x*blockDim.x+threadIdx.x = p.
  // Drops the 64-bit div/mod the 1-D form needed to decode (b, p).
  const int b = blockIdx.y;
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;

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
      // Single f-walk reuses ind_off / p2h loads for both contributions when
      // both flags are live; the no-* arms DCE under the template params.
      for (int f = 0; f < F; ++f) {
        const int64_t off = ind_off[t_F + f];
        const int32_t h_f = policy_to_factor_history[t_F_P + f * P + p];
        if (USE_INDUCTIVE) acc += inductive_b[off + h_f];
        if (USE_FACTOR_SCORES) acc += factor_scores_b[off + h_f];
      }
    }
  }
  out[static_cast<int64_t>(b) * P + p] = acc;
}

cudaError_t launch_final_scatter_dedup(const float* scores_concat, const float* inductive_concat,
                                       const float* factor_scores, const int32_t* policy_to_modality_idx,
                                       const int32_t* policy_to_factor_history, const int64_t* mod_off,
                                       const int64_t* ind_off, int Bn, int T, int M, int F, int P, int64_t total_mod,
                                       int64_t total_ind, bool use_inductive, bool use_factor_scores, float* out,
                                       cudaStream_t stream) {
  if (Bn <= 0 || P <= 0) return cudaSuccess;
  const int64_t total      = static_cast<int64_t>(Bn) * P;
  const int     block_size = launch_block_size_1d(stream, total);
  const dim3    grid(static_cast<unsigned>(launch_blocks(P, block_size)), static_cast<unsigned>(Bn));
  const int     pymdp_b2 = (static_cast<int>(use_inductive) << 1) | static_cast<int>(use_factor_scores);
  switch (pymdp_b2) {
  case 0:
    final_scatter_dedup_kernel<false, false><<<grid, block_size, 0, stream>>>(
        scores_concat, inductive_concat, factor_scores, policy_to_modality_idx, policy_to_factor_history, mod_off,
        ind_off, Bn, T, M, F, P, total_mod, total_ind, out);
    break;
  case 1:
    final_scatter_dedup_kernel<false, true><<<grid, block_size, 0, stream>>>(
        scores_concat, inductive_concat, factor_scores, policy_to_modality_idx, policy_to_factor_history, mod_off,
        ind_off, Bn, T, M, F, P, total_mod, total_ind, out);
    break;
  case 2:
    final_scatter_dedup_kernel<true, false><<<grid, block_size, 0, stream>>>(
        scores_concat, inductive_concat, factor_scores, policy_to_modality_idx, policy_to_factor_history, mod_off,
        ind_off, Bn, T, M, F, P, total_mod, total_ind, out);
    break;
  case 3:
    final_scatter_dedup_kernel<true, true><<<grid, block_size, 0, stream>>>(
        scores_concat, inductive_concat, factor_scores, policy_to_modality_idx, policy_to_factor_history, mod_off,
        ind_off, Bn, T, M, F, P, total_mod, total_ind, out);
    break;
  }
  return cudaGetLastError();
}

// ----------------------------------------------------------------------------
// Device-side A/B/linear repack (learning fast path). See kernels.h for the
// rationale; the gathered per-modality / per-factor slices are D2D
// cudaMemcpy2DAsync issued from cache.cc, so only these two kernels live here.
// ----------------------------------------------------------------------------

// Mirror the host xlogx (logexp_f32.h): x * logf(max(x, kLogEps)), kLogEps=1e-12.
__device__ __forceinline__ float dev_xlogx(float x) {
  return x * logf(fmaxf(x, 1e-12f));
}

// dst[b, (o*S_split+s)*K_keep + k] = packed[b, (o*K_keep+k)*S_split + s].
// One thread per output element. Output per-batch layout is [O_m, S_split, K_keep].
__global__ void a_rank3_cublas_view_kernel(const float* __restrict__ packed, int Bn, int O_m, int K_keep, int S_split,
                                           float* __restrict__ dst) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_b = static_cast<int64_t>(O_m) * S_split * K_keep;
  const int64_t total = static_cast<int64_t>(Bn) * per_b;
  if (idx >= total) return;

  const int     b  = static_cast<int>(idx / per_b);
  int64_t       r  = idx - static_cast<int64_t>(b) * per_b;
  const int64_t sk = static_cast<int64_t>(S_split) * K_keep;
  const int     o  = static_cast<int>(r / sk);
  r -= static_cast<int64_t>(o) * sk;
  const int s = static_cast<int>(r / K_keep);
  const int k = static_cast<int>(r - static_cast<int64_t>(s) * K_keep);

  const size_t src_off = static_cast<size_t>(b) * per_b + (static_cast<size_t>(o) * K_keep + k) * S_split + s;
  dst[idx]             = packed[src_off];
}

cudaError_t launch_a_rank3_cublas_view(const float* packed, int Bn, int O_m, int K_keep, int S_split, float* dst,
                                       cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * O_m * S_split * K_keep,
                  a_rank3_cublas_view_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                      packed, Bn, O_m, K_keep, S_split, dst));
}

// dst[b, k] = (use_utility ? sum_o A_g[b, o, k] * C_btm[o] : 0)
//           + (use_states_info_gain ? sum_o xlogx(A_g[b, o, k]) : 0)
// A_g per-batch layout [O, K_m]; C_btm = C + b*C_off_M + C_off_m + t*O, length O.
__global__ void linear_precompute_tm_kernel(const float* __restrict__ A_g, const float* __restrict__ C, int Bn, int O,
                                            int K_m, int64_t C_off_M, int64_t C_off_m, int t, bool use_utility,
                                            bool use_states_info_gain, float* __restrict__ dst) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * K_m;
  if (idx >= total) return;

  const int    b     = static_cast<int>(idx / K_m);
  const int    k     = static_cast<int>(idx - static_cast<int64_t>(b) * K_m);
  const float* A_bk  = A_g + static_cast<size_t>(b) * O * K_m + k;  // stride K_m over o
  const float* C_btm = C + static_cast<size_t>(b) * C_off_M + C_off_m + static_cast<int64_t>(t) * O;

  float acc = 0.0f;
  for (int o = 0; o < O; ++o) {
    const float a = A_bk[static_cast<size_t>(o) * K_m];
    if (use_utility) acc += a * C_btm[o];
    if (use_states_info_gain) acc += dev_xlogx(a);
  }
  dst[idx] = acc;
}

cudaError_t launch_linear_precompute_tm(const float* A_g, const float* C, int Bn, int O, int K_m, int64_t C_off_M,
                                        int64_t C_off_m, int t, bool use_utility, bool use_states_info_gain, float* dst,
                                        cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * K_m,
                  linear_precompute_tm_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                      A_g, C, Bn, O, K_m, C_off_M, C_off_m, t, use_utility, use_states_info_gain, dst));
}

// ----------------------------------------------------------------------------
// Device wA/wB (param-info-gain) repack. Mirrors precompute_wA /
// precompute_wB_transposed (neg_efe_precompute.h) so the learning + novelty
// path rebuilds the Dirichlet weights on device — no host digamma + round-trip.
// ----------------------------------------------------------------------------

// pymdp.maths.MINVAL = std::numeric_limits<float>::epsilon() (= 2^-23).
constexpr float kDevWnormMinval = 1.1920929e-7f;

// Device port of digamma_f32 (common/kernel_primitives.h): recurrence to x >= 6
// then the asymptotic Bernoulli expansion. Bit-pattern non-finite check so it
// survives -ffast-math/--use_fast_math, matching the host.
__device__ __forceinline__ float dev_digamma_f32(float x) {
  const unsigned int bits = __float_as_uint(x);
  if (!(x > 0.0f) || (bits & 0x7f800000u) == 0x7f800000u) return -1e30f;
  float result = 0.0f;
  int   steps  = 0;
  while (x < 6.0f && steps < 64) {
    result -= 1.0f / x;
    x += 1.0f;
    ++steps;
  }
  const float x2 = x * x;
  const float x4 = x2 * x2;
  const float x6 = x4 * x2;
  const float x8 = x4 * x4;
  result +=
      logf(x) - 0.5f / x - 1.0f / (12.0f * x2) + 1.0f / (120.0f * x4) - 1.0f / (252.0f * x6) + 1.0f / (240.0f * x8);
  return result;
}

// w(p; sum) = log(sum) - log(p) + 1/p - 1/sum + digamma(p) - digamma(sum); 0 for
// p <= 0 (the JAX (p > 0) mask). The (log/inv/dig)_sum triple is the per-cell
// summary of the safe column sum, computed once by the caller.
__device__ __forceinline__ float dev_wnorm_weight(float p, float log_sum, float inv_sum, float dig_sum) {
  if (p <= 0.0f) return 0.0f;
  const float ps = fmaxf(p, kDevWnormMinval);
  return log_sum - logf(ps) + 1.0f / ps - inv_sum + dev_digamma_f32(ps) - dig_sum;
}

// wA for one modality. pA per-batch layout [O, K_m] (stride K_m over o), strided
// by pA_batch_stride; one thread per (b, k) folds the column sum then writes the
// O entries. Output [O, K_m] matches A's packed layout.
__global__ void wnorm_a_kernel(const float* __restrict__ pA, int64_t pA_batch_stride, int64_t pA_mod_off, int Bn, int O,
                               int K_m, float* __restrict__ wA) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * K_m;
  if (idx >= total) return;

  const int    b   = static_cast<int>(idx / K_m);
  const int    k   = static_cast<int>(idx - static_cast<int64_t>(b) * K_m);
  const float* pAb = pA + static_cast<int64_t>(b) * pA_batch_stride + pA_mod_off + k;  // stride K_m over o
  float*       wAb = wA + static_cast<int64_t>(b) * O * K_m + k;

  float sum = 0.0f;
  for (int o = 0; o < O; ++o) sum += pAb[static_cast<int64_t>(o) * K_m];
  const float ss      = fmaxf(sum, kDevWnormMinval);
  const float log_sum = logf(ss);
  const float inv_sum = 1.0f / ss;
  const float dig_sum = dev_digamma_f32(ss);
  for (int o = 0; o < O; ++o) {
    wAb[static_cast<int64_t>(o) * K_m] =
        dev_wnorm_weight(pAb[static_cast<int64_t>(o) * K_m], log_sum, inv_sum, dig_sum);
  }
}

cudaError_t launch_wnorm_a(const float* pA, int64_t pA_batch_stride, int64_t pA_mod_off, int Bn, int O, int K_m,
                           float* wA, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * K_m, wnorm_a_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(
                                                      pA, pA_batch_stride, pA_mod_off, Bn, O, K_m, wA));
}

// wB for one factor. pB per-batch layout [S, K, U] (pB[(s*K+k)*U + u]); the
// column sum is over S. One thread per (b, k, u) writes the transposed (U, S, K)
// output wB[(u*S+s)*K + k], matching precompute_wB_transposed / pack_b_factors.
__global__ void wnorm_b_kernel(const float* __restrict__ pB, int64_t pB_batch_stride, int64_t pB_fac_off, int Bn, int S,
                               int K, int U, float* __restrict__ wB) {
  const int64_t idx   = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t total = static_cast<int64_t>(Bn) * K * U;
  if (idx >= total) return;

  const int     b   = static_cast<int>(idx / (static_cast<int64_t>(K) * U));
  const int64_t r   = idx - static_cast<int64_t>(b) * K * U;
  const int     k   = static_cast<int>(r / U);
  const int     u   = static_cast<int>(r - static_cast<int64_t>(k) * U);
  const float*  pBb = pB + static_cast<int64_t>(b) * pB_batch_stride + pB_fac_off;
  float*        wBb = wB + static_cast<int64_t>(b) * S * K * U;

  float sum = 0.0f;
  for (int s = 0; s < S; ++s) sum += pBb[(static_cast<int64_t>(s) * K + k) * U + u];
  const float ss      = fmaxf(sum, kDevWnormMinval);
  const float log_sum = logf(ss);
  const float inv_sum = 1.0f / ss;
  const float dig_sum = dev_digamma_f32(ss);
  for (int s = 0; s < S; ++s) {
    const float p                                  = pBb[(static_cast<int64_t>(s) * K + k) * U + u];
    wBb[(static_cast<int64_t>(u) * S + s) * K + k] = dev_wnorm_weight(p, log_sum, inv_sum, dig_sum);
  }
}

cudaError_t launch_wnorm_b(const float* pB, int64_t pB_batch_stride, int64_t pB_fac_off, int Bn, int S, int K, int U,
                           float* wB, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(static_cast<int64_t>(Bn) * K * U,
                  wnorm_b_kernel<<<pymdp_launch_blocks, pymdp_block_size, 0, stream>>>(pB, pB_batch_stride, pB_fac_off,
                                                                                       Bn, S, K, U, wB));
}

}  // namespace cuda_kernels
}  // namespace pymdp_ffi
