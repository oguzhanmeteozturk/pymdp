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

// =============================================================================
// Common helpers
// =============================================================================

// Default block size. Kernels with different register pressure can override it.
constexpr int kBlockSize = 128;

__host__ __device__ __forceinline__ int idiv_ceil(int a, int b) {
  return (a + b - 1) / b;
}
inline int launch_blocks(int total) {
  return idiv_ceil(total, kBlockSize);
}

// Host-only 1-D grid launch. The variadic body must be a kernel launch using
// `pymdp_launch_blocks` as the grid dimension (e.g.
// my_kernel<<<pymdp_launch_blocks, kBlockSize, 0, stream>>>(...)).
//
// Variadic on purpose: nvcc 10.2's preprocessor treats every comma at the
// macro-argument level as an argument separator, so a non-variadic
// `(total, stmt)` form would split a `kernel<<<...>>>(a, b, c)` call into
// many macro arguments. Funneling everything past `total` through __VA_ARGS__
// preserves the kernel launch syntax.
//
// IMPORTANT: This macro `return`s from the enclosing function (cudaSuccess on
// empty grid, otherwise cudaGetLastError() after the launch). It must be the
// terminal statement of a `cudaError_t`-returning launcher; do not nest it
// inside `if`/`for` or follow it with more code.
#define PYMDP_LAUNCH_1D(total, ...)                                                                                    \
  do {                                                                                                                 \
    const int pymdp_launch_total = (total);                                                                            \
    if (pymdp_launch_total <= 0) return cudaSuccess;                                                                   \
    const int pymdp_launch_blocks = launch_blocks(pymdp_launch_total);                                                 \
    __VA_ARGS__;                                                                                                       \
    return cudaGetLastError();                                                                                         \
  } while (0)

// Switch on values 1..8 inclusive. BODY(N) must expand to `case N: ... break;`.
//
// IMPORTANT: The `default:` arm `return`s cudaErrorInvalidValue, so this
// macro must appear inside a `cudaError_t`-returning function. Place it
// where falling through the switch is the intended success path.
#define PYMDP_CUDA_DISPATCH_1_8(expr, BODY)                                                                            \
  switch (expr) {                                                                                                      \
    BODY(1)                                                                                                            \
    BODY(2) BODY(3) BODY(4) BODY(5) BODY(6) BODY(7) BODY(8) default : return cudaErrorInvalidValue;                    \
  }

// Lift the (use_states, use_linear, use_pA) runtime flags into compile-time
// template parameters so the device compiler can DCE the dead arms, drop the
// register slots backing them, and unroll inner loops more aggressively. The
// per-launch cost is one 8-way switch on the host. BODY(S, L, P) must be a
// statement that uses S, L, P as constexpr bool tokens.
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

// =============================================================================
// Generalized B-rollout kernel for factor-local or multi-parent B-deps.
//
// Computes (per factor f, level t):
//   qs_outer[b, h, k] = prod_i parents.qs[i][b, parent_histories[h*N_PARENTS + i], s_i(k)]
//   qs_out[b, h, s]   = sum_k B[b, s, k, action[h]] * qs_outer[b, h, k]
// where K_f = prod_i parents.S[i] and k decomposes into (s_0, ...,
// s_{N_PARENTS-1}) row-major (last innermost).
//
// One block per (b, h_next). Threads cooperatively build qs_outer[K_f] in
// shared memory, sync, then compute qs_out[s] via dot product against
// B[b, s, :, u] (s strided across threads). When wB_tr / factor_score are
// supplied (param-info-gain mode), the same parallel pass also accumulates
// each thread's contribution to the wB novelty score and a block reduction
// sums to a single factor_score per (b, h).
//
// Shared memory ceiling: K_f floats per block. Maxwell sm_53 has 48 KB / SM,
// so K_f must be <= 12288. Past that the launch fails with
// cudaErrorInvalidConfiguration and you'd need to chunk the K loop.
//
// Per-N specialization lives inside the templated kernel as compile-time
// branches on N_PARENTS (the dead arms DCE). N_PARENTS in {1, 2, 3} hoist
// the per-parent qs/H/S into named __restrict__ locals at entry and use an
// explicit phase-1 body with inline divmods. N_PARENTS >= 4 takes an
// indexed Kronecker loop through the BRolloutParents struct. Adding a
// hand-tuned N=4 path is one new `else if` arm.
// =============================================================================

// Block-wide sum reduction over kBlockSize threads. All threads receive the
// same total (every thread reads buf[0] at exit). The static __shared__
// buffer (512 B for kBlockSize=128) merges into the calling kernel's shmem
// pool via __forceinline__ and coexists with the kernel's dynamic qs_outer
// extern. Sequential-addressing tree, portable across sm_53 and sm_86.
__device__ __forceinline__ float b_rollout_block_reduce_sum(float val) {
  __shared__ float buf[kBlockSize];
  buf[threadIdx.x] = val;
  __syncthreads();
#pragma unroll
  for (int stride = kBlockSize >> 1; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      buf[threadIdx.x] += buf[threadIdx.x + stride];
    }
    __syncthreads();
  }
  return buf[0];
}

// Fused phase 2 (per-s dot against B → qs_out) + factor_score reduction
// (per-thread partial of qs_out[s] * sum_k wB[s, k] * qs_outer[k], summed
// block-wide). Hoisted into a __device__ helper so the per-N phase-1 arms
// stay tight; __forceinline__ folds it back into the kernel so per-N
// constant propagation isn't broken.
__device__ __forceinline__ void b_rollout_phase2_and_wb(const float* __restrict__ B, const float* __restrict__ wB_tr,
                                                        int b, int h, int u, int Nh, int S_f, int K_f, int U_f,
                                                        const float* __restrict__ qs_outer, float* __restrict__ qs_out,
                                                        float* __restrict__ factor_score) {
  // Per-block bases hoisted so the 64-bit indexing math (b * S_f * row_stride_B
  // and (b * Nh + h) * S_f) is computed once; the per-s / per-k inner work
  // then runs in int32 with pointer-arith widening at the back end.
  const int    row_stride_B = K_f * U_f;
  const float* B_b          = B + static_cast<size_t>(b) * S_f * row_stride_B;
  const size_t score_off    = static_cast<size_t>(b) * Nh + h;
  float* const qs_out_bh    = qs_out + score_off * S_f;

  // `compute_wb` is uniform across the block (wB_tr / factor_score are
  // kernel args), so the per-s `if (compute_wb)` doesn't diverge warps and
  // the compiler hoists wb_bu out of the per-s loop.
  const bool   compute_wb = (wB_tr != nullptr && factor_score != nullptr);
  const float* wb_bu      = compute_wb ? wB_tr + (static_cast<size_t>(b) * U_f + u) * S_f * K_f : nullptr;

  // Per-thread partial of factor_score = sum_s (qs_out[s] * sum_k
  // wB[s, k] * qs_outer[k]). Threads with no s assigned contribute 0;
  // threads with multiple s-stripes accumulate over them.
  float partial = 0.0f;

  // Phase 2 + fused factor_score per-s contribution.
  // B[b, s, k, u] at b*S_f*K_f*U_f + s*K_f*U_f + k*U_f + u.
  for (int s = threadIdx.x; s < S_f; s += blockDim.x) {
    const float* Bf   = B_b + s * row_stride_B;
    const float* wb_s = compute_wb ? (wb_bu + s * K_f) : nullptr;
    float        acc  = 0.0f;
    float        wacc = 0.0f;
    // Fused k-loop: shares the qs_outer[k] load between the B and wB
    // contractions when wB is live.
    for (int k = 0; k < K_f; ++k) {
      const float qok = qs_outer[k];
      acc += Bf[k * U_f + u] * qok;
      if (compute_wb) wacc += wb_s[k] * qok;
    }
    qs_out_bh[s] = acc;
    if (compute_wb) partial += acc * wacc;
  }

  if (compute_wb) {
    // `partial` is a per-thread register — no inter-thread dependency on
    // phase 2's qs_out_bh writes, so no barrier is needed before the
    // reduction. qs_out_bh visibility to downstream kernels is handled by
    // the implicit kernel-exit barrier.
    const float total = b_rollout_block_reduce_sum(partial);
    if (threadIdx.x == 0) {
      factor_score[score_off] = total;
    }
  } else if (factor_score != nullptr && threadIdx.x == 0) {
    // wB absent but caller still wants the slot zeroed so the final
    // scatter sees a well-defined value.
    factor_score[score_off] = 0.0f;
  }
}

template <int N_PARENTS>
__global__ void b_rollout_general_kernel(const float* __restrict__ B, const float* __restrict__ wB_tr,
                                         const int32_t* __restrict__ action_h,
                                         const int32_t* __restrict__ parent_histories, BRolloutParents parents, int Bn,
                                         int Nh, int S_f, int K_f, int U_f, float* __restrict__ qs_out,
                                         float* __restrict__ factor_score) {
  const int b = blockIdx.x;
  const int h = blockIdx.y;
  if (b >= Bn || h >= Nh) return;

  extern __shared__ float qs_outer[];

  const int u = action_h[h];

  // Phase 1: build qs_outer[K_f] = prod_i parents.qs[i][b, parent_h_i, s_i(k)].
  // The if-cascade keys on N_PARENTS (a template parameter), so each
  // instantiation keeps exactly one arm — the others DCE.
  if (N_PARENTS == 1) {
    const float* __restrict__ qs0 = parents.qs[0];
    const int    H0               = parents.H[0];
    const int    S0               = parents.S[0];
    const int    parent_h0        = parent_histories[h];
    const float* qs0_bh           = qs0 + (static_cast<size_t>(b) * H0 + parent_h0) * S0;
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      qs_outer[k] = qs0_bh[k];
    }
  } else if (N_PARENTS == 2) {
    const float* __restrict__ qs0 = parents.qs[0];
    const float* __restrict__ qs1 = parents.qs[1];
    const int    H0               = parents.H[0];
    const int    H1               = parents.H[1];
    const int    S0               = parents.S[0];
    const int    S1               = parents.S[1];
    const int    parent_h0        = parent_histories[h * 2 + 0];
    const int    parent_h1        = parent_histories[h * 2 + 1];
    const float* qs0_bh           = qs0 + (static_cast<size_t>(b) * H0 + parent_h0) * S0;
    const float* qs1_bh           = qs1 + (static_cast<size_t>(b) * H1 + parent_h1) * S1;
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      const int s_0 = k / S1;
      const int s_1 = k - s_0 * S1;
      qs_outer[k]   = qs0_bh[s_0] * qs1_bh[s_1];
    }
  } else if (N_PARENTS == 3) {
    const float* __restrict__ qs0 = parents.qs[0];
    const float* __restrict__ qs1 = parents.qs[1];
    const float* __restrict__ qs2 = parents.qs[2];
    const int    H0               = parents.H[0];
    const int    H1               = parents.H[1];
    const int    H2               = parents.H[2];
    const int    S0               = parents.S[0];
    const int    S1               = parents.S[1];
    const int    S2               = parents.S[2];
    const int    parent_h0        = parent_histories[h * 3 + 0];
    const int    parent_h1        = parent_histories[h * 3 + 1];
    const int    parent_h2        = parent_histories[h * 3 + 2];
    const float* qs0_bh           = qs0 + (static_cast<size_t>(b) * H0 + parent_h0) * S0;
    const float* qs1_bh           = qs1 + (static_cast<size_t>(b) * H1 + parent_h1) * S1;
    const float* qs2_bh           = qs2 + (static_cast<size_t>(b) * H2 + parent_h2) * S2;
    const int    S_12             = S1 * S2;
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      const int s_0 = k / S_12;
      int       rem = k - s_0 * S_12;
      const int s_1 = rem / S2;
      const int s_2 = rem - s_1 * S2;
      qs_outer[k]   = qs0_bh[s_0] * qs1_bh[s_1] * qs2_bh[s_2];
    }
  } else {
    // Generic N_PARENTS >= 4: indexed Kronecker through the parents struct.
    int parent_h_local[N_PARENTS];
#pragma unroll
    for (int i = 0; i < N_PARENTS; ++i) {
      parent_h_local[i] = parent_histories[h * N_PARENTS + i];
    }
    int strides[N_PARENTS];
    strides[N_PARENTS - 1] = 1;
#pragma unroll
    for (int i = N_PARENTS - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * parents.S[i + 1];
    }
    for (int k = threadIdx.x; k < K_f; k += blockDim.x) {
      int   rem = k;
      float val = 1.0f;
#pragma unroll
      for (int i = 0; i < N_PARENTS; ++i) {
        const int s_i = (i + 1 < N_PARENTS) ? (rem / strides[i]) : rem;
        rem -= s_i * strides[i];
        const int H_i = parents.H[i];
        const int S_i = parents.S[i];
        val *= parents.qs[i][((b * H_i) + parent_h_local[i]) * S_i + s_i];
      }
      qs_outer[k] = val;
    }
  }
  __syncthreads();

  b_rollout_phase2_and_wb(B, wB_tr, b, h, u, Nh, S_f, K_f, U_f, qs_outer, qs_out, factor_score);
}

namespace {

template <int N_PARENTS>
inline void launch_b_rollout_tpl(const float* B, const float* wB_tr, const int32_t* action_h,
                                 const int32_t* parent_histories, BRolloutParents parents, int Bn, int Nh, int S_f,
                                 int K_f, int U_f, float* qs_out, float* factor_score, cudaStream_t stream) {
  // Empty grid — dim3(0, ...) trips cudaErrorInvalidConfiguration on some
  // drivers, so short-circuit before constructing the launch.
  if (Bn <= 0 || Nh <= 0) return;
  const dim3   grid(static_cast<unsigned>(Bn), static_cast<unsigned>(Nh), 1);
  const size_t shmem_bytes = static_cast<size_t>(K_f) * sizeof(float);
  b_rollout_general_kernel<N_PARENTS><<<grid, kBlockSize, shmem_bytes, stream>>>(
      B, wB_tr, action_h, parent_histories, parents, Bn, Nh, S_f, K_f, U_f, qs_out, factor_score);
}

}  // namespace

cudaError_t launch_b_rollout_general(const float* B, const float* wB_tr, const int32_t* action_h,
                                     const int32_t* parent_histories, const BRolloutParents& parents, int n_parents,
                                     int Bn, int Nh, int S_f, int K_f, int U_f, float* qs_out, float* factor_score,
                                     cudaStream_t stream) {
#define LAUNCH_B_ROLLOUT(N)                                                                                            \
  case N:                                                                                                              \
    launch_b_rollout_tpl<N>(B, wB_tr, action_h, parent_histories, parents, Bn, Nh, S_f, K_f, U_f, qs_out,              \
                            factor_score, stream);                                                                     \
    break;
  PYMDP_CUDA_DISPATCH_1_8(n_parents, LAUNCH_B_ROLLOUT)
#undef LAUNCH_B_ROLLOUT
  return cudaGetLastError();
}

// =============================================================================
// Common scoring helpers
// =============================================================================

// Optional explicit read-only loads via __ldg(). On Maxwell the read-only
// data cache is separate from L1, which can help some memory-bound shapes.
//
// Gated on __CUDA_ARCH__ inside the helper because nvcc runs the .cu
// source through both the host and device front-ends; bare __ldg
// references inside __global__ bodies still get parsed by the host pass
// and trip "identifier __ldg is undefined" errors there. The wrapper is
// also gated on PYMDP_FFI_CUDA_LDG (opt-in) because the main cuBLAS path does
// not benefit consistently. Keep the build flag for shape regimes where
// memory bandwidth is dominant.
#ifdef PYMDP_FFI_CUDA_LDG
template <typename T> __host__ __device__ __forceinline__ T ro_load(const T* p) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)
  return __ldg(p);
#else
  return *p;
#endif
}
#define LDG(x) ro_load(&(x))
#else
#define LDG(x) (x)
#endif

// Fast log for the entropy-term xlogx clamp. __logf is the single-precision
// intrinsic and is faster on Maxwell for these in-range values.
// Default on; -DPYMDP_FFI_CUDA_NO_FAST_LOG falls back to logf.
__device__ __forceinline__ float clamp_log(float v) {
  // Matches CPU kLogEps. log(max(v, 1e-16)) for the entropy clamp.
  v = fmaxf(v, 1e-16f);
#ifndef PYMDP_FFI_CUDA_NO_FAST_LOG
  return v * __logf(v);
#else
  return v * logf(v);
#endif
}

template <bool USE_STATES>
__device__ __forceinline__ void write_modality_score(float* __restrict__ score_out, int64_t b, int64_t b_stride,
                                                     int64_t local_h, float acc_ent, float linear_acc) {
  score_out[b * b_stride + local_h] = (USE_STATES ? -acc_ent : 0.0f) + linear_acc;
}

// =============================================================================
// Rank-1 dedup modality scoring.
//
// score_out[b, h_d_0] = (use_states ? -sum_o xlogx(qo[b, h_d_0, o]) : 0)
//                     + (use_linear ? sum_{s_0} linear[b, s_0] * qs[b, h_d_0, s_0] : 0)
//                     + (use_pA   ? sum_o qo[o] * (sum_{s_0} wA[b, o, s_0] * qs[b, h_d_0, s_0]) : 0)
// where qo[b, h_d_0, o] = sum_{s_0} A[b, o, s_0] * qs[b, h_d_0, s_0].
//
// One thread per (b, h_d_0). Per-thread work is O*S_d_0 for qo, plus
// O*S_d_0 for the optional pA novelty pass, plus S_d_0 for the optional
// linear term.
// =============================================================================

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank1_kernel(const float* __restrict__ A, const float* __restrict__ wA,
                                                  const float* __restrict__ linear, const float* __restrict__ qs_d_0,
                                                  int Bn, int O, int H_d_0, int S_d_0, int64_t b_stride,
                                                  float* __restrict__ score_out) {
  int idx   = blockIdx.x * blockDim.x + threadIdx.x;
  int total = Bn * H_d_0;
  if (idx >= total) return;
  int b = idx / H_d_0;
  int h = idx % H_d_0;

  const float* A_b      = A + b * O * S_d_0;
  const float* wA_b     = USE_PA ? (wA + b * O * S_d_0) : nullptr;
  const float* qs_p     = qs_d_0 + (b * H_d_0 + h) * S_d_0;
  const float* linear_b = USE_LINEAR ? (linear + b * S_d_0) : nullptr;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
    for (int o = 0; o < O; ++o) {
      const float* A_bo = A_b + o * S_d_0;
      float        qo_o = 0.0f;
      for (int i = 0; i < S_d_0; ++i) qo_o += LDG(A_bo[i]) * LDG(qs_p[i]);
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) {
        const float* wA_bo = wA_b + o * S_d_0;
        float        wa_o  = 0.0f;
        for (int i = 0; i < S_d_0; ++i) wa_o += LDG(wA_bo[i]) * LDG(qs_p[i]);
        pA_acc += qo_o * wa_o;
      }
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int i = 0; i < S_d_0; ++i) linear_acc += LDG(linear_b[i]) * LDG(qs_p[i]);
  }

  // score_out points to the (t, m) slice within scores_concat (shape
  // [Bn, total_mod_entries]); b-stride therefore = total_mod_entries.
  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

cudaError_t launch_modality_score_dedup_rank1(const float* A_unflat, const float* wA_unflat, const float* linear,
                                              const float* qs_d_0, int Bn, int O, int H_d_0, int S_d_0,
                                              int64_t b_stride, bool use_states, bool use_linear, bool use_pA,
                                              float* score_out, cudaStream_t stream) {
  const int total = Bn * H_d_0;
  if (total <= 0) return cudaSuccess;
  const int blocks = launch_blocks(total);
#define LAUNCH_R1(S, L, P)                                                                                             \
  modality_score_dedup_rank1_kernel<S, L, P><<<blocks, kBlockSize, 0, stream>>>(                                       \
      A_unflat, wA_unflat, linear, qs_d_0, Bn, O, H_d_0, S_d_0, b_stride, score_out)
  PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, LAUNCH_R1);
#undef LAUNCH_R1
  return cudaGetLastError();
}

// =============================================================================
// Tiny-shape fused rank-2 / rank-3 modality scoring.
//
// These kernels deliberately bypass the q01_outer + cuBLAS helper pipeline for
// small shapes. They trade more direct scalar work per output history tuple for
// fewer kernel launches, no q01_outer materialization, no tmp_qo/tmp_wa/tmp_lin,
// and no rank-3 cublas-layout transpose.
//
// Intended for Jetson Nano / sm_53 tiny active-inference shapes. The host-side
// heuristic decides when to use these; the existing cuBLAS path remains the
// fallback for larger shapes.
// =============================================================================

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_fused_tiny_kernel(const float* __restrict__ A, const float* __restrict__ wA,
                                             const float* __restrict__ linear, const float* __restrict__ qs_d_0,
                                             const float* __restrict__ qs_d_1, int Bn, int O, int H_d_0, int H_d_1,
                                             int S_d_0, int S_d_1, int64_t b_stride, float* __restrict__ score_out) {
  const int H_kk  = H_d_0 * H_d_1;
  const int K_d   = S_d_0 * S_d_1;
  const int total = Bn * H_kk;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int b  = idx / H_kk;
  const int h  = idx - b * H_kk;
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
        const float q0 = LDG(qs0[s0]);
        for (int s1 = 0; s1 < S_d_1; ++s1, ++k) {
          const float q = q0 * LDG(qs1[s1]);
          qo_o += LDG(A_bo[k]) * q;
          if (USE_PA) wa_o += LDG(wA_bo[k]) * q;
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
      const float q0 = LDG(qs0[s0]);
      for (int s1 = 0; s1 < S_d_1; ++s1, ++k) {
        linear_acc += LDG(lin_b[k]) * q0 * LDG(qs1[s1]);
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
  const int total = Bn * H_d_0 * H_d_1;
  if (total <= 0) return cudaSuccess;
  const int blocks = launch_blocks(total);
#define LAUNCH_R2_TINY(S, L, P)                                                                                        \
  modality_score_dedup_rank2_fused_tiny_kernel<S, L, P><<<blocks, kBlockSize, 0, stream>>>(                            \
      A_unflat, wA_unflat, linear, qs_d_0, qs_d_1, Bn, O, H_d_0, H_d_1, S_d_0, S_d_1, b_stride, score_out)
  PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, LAUNCH_R2_TINY);
#undef LAUNCH_R2_TINY
  return cudaGetLastError();
}

template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void modality_score_dedup_rank3_fused_tiny_kernel(
    const float* __restrict__ A, const float* __restrict__ wA, const float* __restrict__ linear,
    const float* __restrict__ qs_d_0, const float* __restrict__ qs_d_1, const float* __restrict__ qs_d_2, int Bn, int O,
    int H_d_0, int H_d_1, int H_d_2, int S_d_0, int S_d_1, int S_d_2, int64_t b_stride, float* __restrict__ score_out) {
  const int H_12  = H_d_1 * H_d_2;
  const int H_all = H_d_0 * H_12;
  const int K_d   = S_d_0 * S_d_1 * S_d_2;
  const int total = Bn * H_all;

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total) return;

  const int b    = idx / H_all;
  int       rest = idx - b * H_all;
  const int h0   = rest / H_12;
  rest -= h0 * H_12;
  const int h1 = rest / H_d_2;
  const int h2 = rest - h1 * H_d_2;

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
        const float q0 = LDG(qs0[s0]);
        for (int s1 = 0; s1 < S_d_1; ++s1) {
          const float q01 = q0 * LDG(qs1[s1]);
          for (int s2 = 0; s2 < S_d_2; ++s2, ++k) {
            const float q = q01 * LDG(qs2[s2]);
            qo_o += LDG(A_bo[k]) * q;
            if (USE_PA) wa_o += LDG(wA_bo[k]) * q;
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
      const float q0 = LDG(qs0[s0]);
      for (int s1 = 0; s1 < S_d_1; ++s1) {
        const float q01 = q0 * LDG(qs1[s1]);
        for (int s2 = 0; s2 < S_d_2; ++s2, ++k) {
          linear_acc += LDG(lin_b[k]) * q01 * LDG(qs2[s2]);
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
  const int total = Bn * H_d_0 * H_d_1 * H_d_2;
  if (total <= 0) return cudaSuccess;
  const int blocks = launch_blocks(total);
#define LAUNCH_R3_TINY(S, L, P)                                                                                        \
  modality_score_dedup_rank3_fused_tiny_kernel<S, L, P>                                                                \
      <<<blocks, kBlockSize, 0, stream>>>(A_unflat, wA_unflat, linear, qs_d_0, qs_d_1, qs_d_2, Bn, O, H_d_0, H_d_1,    \
                                          H_d_2, S_d_0, S_d_1, S_d_2, b_stride, score_out)
  PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, LAUNCH_R3_TINY);
#undef LAUNCH_R3_TINY
  return cudaGetLastError();
}

// =============================================================================
// cuBLAS helper kernels used by rank-2 and rank-3 modality scoring. The .cc
// TU launches these helpers around the cublasSgemmStridedBatched call.
//
// Rank-3 stage 1 pipeline (per (t, m=rank3) per b):
//   1. build_qs01_outer_kernel: q01[b, k_keep, h_kk] = qs_keep_0[h_0, s_0] *
//                                qs_keep_1[h_1, s_1]
//   2. cublasSgemmStridedBatched (called from .cc): tmp_qo_cublas[b,
//      o*S_split, h_kk] = sum_{k_keep} A_cublas[b, o*S_split, k_keep] *
//      q01[b, k_keep, h_kk]
//   3. tmp_qo_cublas_to_my_kernel: transposes from [b, o*S_split, h_kk] to
//      stage-2's expected [b, h_kk, o, s_split] layout.
//   4. tmp_lin_per_h_kernel: per-(b, h_kk, s_split) thread computes
//      sum_{k_keep} q01 * linear -> tmp_lin.
// Rank-3 stage 2 (modality_score_dedup_rank3_split_stage2_kernel below)
// then folds in entropy + linear.
//
// Rank-2 pipeline reuses build_qs01_outer + cublasSgemmStridedBatched
// (with no s_split dimension) and finishes via
// modality_score_dedup_rank2_cublas_finish_kernel below.
// =============================================================================

// Build q01[b, k_keep, h_kk] = qs_keep_0[b, h_0, s_0] * qs_keep_1[b, h_1, s_1].
// One thread per (b, k_keep, h_kk) output element. Layout matches what the
// downstream cuBLAS GEMM expects for its B operand.
__global__ void build_qs01_outer_kernel(const float* __restrict__ qs_keep_0, const float* __restrict__ qs_keep_1,
                                        int Bn, int H_0, int H_1, int S_0, int S_1, float* __restrict__ q01_outer) {
  int       idx    = blockIdx.x * blockDim.x + threadIdx.x;
  const int K_keep = S_0 * S_1;
  const int H_kk   = H_0 * H_1;
  const int total  = Bn * K_keep * H_kk;
  if (idx >= total) return;

  const int b   = idx / (K_keep * H_kk);
  int       r   = idx - b * (K_keep * H_kk);
  const int k   = r / H_kk;
  const int h   = r - k * H_kk;
  const int s_0 = k / S_1;
  const int s_1 = k - s_0 * S_1;
  const int h_0 = h / H_1;
  const int h_1 = h - h_0 * H_1;

  const float qs0 = qs_keep_0[(b * H_0 + h_0) * S_0 + s_0];
  const float qs1 = qs_keep_1[(b * H_1 + h_1) * S_1 + s_1];
  q01_outer[idx]  = qs0 * qs1;
}

cudaError_t launch_build_qs01_outer(const float* qs_keep_0, const float* qs_keep_1, int Bn, int H_0, int H_1, int S_0,
                                    int S_1, float* q01_outer, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(Bn * S_0 * S_1 * H_0 * H_1, build_qs01_outer_kernel<<<pymdp_launch_blocks, kBlockSize, 0, stream>>>(
                                                  qs_keep_0, qs_keep_1, Bn, H_0, H_1, S_0, S_1, q01_outer));
}

// Transpose tmp_qo_cublas[Bn, O*S_split, H_kk] to
// split_tmp_qo[Bn, H_kk, O, S_split].
// One thread per output element. The finish kernel reads the latter layout
// (s_split inner-most for cache-friendly per-(b,h,o) striding).
__global__ void tmp_qo_cublas_to_my_kernel(const float* __restrict__ tmp_cublas, int Bn, int O, int S_split, int H_kk,
                                           float* __restrict__ tmp_my) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_b = H_kk * O * S_split;
  const int total = Bn * per_b;
  if (idx >= total) return;

  const int b = idx / per_b;
  int       r = idx - b * per_b;
  const int h = r / (O * S_split);
  r -= h * (O * S_split);
  const int o = r / S_split;
  const int s = r - o * S_split;

  // tmp_cublas[b, o*S_split + s, h] to tmp_my[b, h, o, s].
  const size_t cublas_off = static_cast<size_t>(b) * O * S_split * H_kk + (o * S_split + s) * H_kk + h;
  tmp_my[idx]             = tmp_cublas[cublas_off];
}

cudaError_t launch_tmp_qo_cublas_to_my(const float* tmp_cublas, int Bn, int O, int S_split, int H_kk, float* tmp_my,
                                       cudaStream_t stream) {
  PYMDP_LAUNCH_1D(Bn * H_kk * O * S_split, tmp_qo_cublas_to_my_kernel<<<pymdp_launch_blocks, kBlockSize, 0, stream>>>(
                                               tmp_cublas, Bn, O, S_split, H_kk, tmp_my));
}

// Compute tmp_lin[b, h, s_split] = sum_{k_keep} q01_outer[b, k_keep, h] *
// linear[b, k_keep * S_split + s_split]. One thread per output.
__global__ void tmp_lin_per_h_kernel(const float* __restrict__ q01_outer, const float* __restrict__ linear, int Bn,
                                     int K_keep, int H_kk, int S_split, float* __restrict__ tmp_lin) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int per_b = H_kk * S_split;
  const int total = Bn * per_b;
  if (idx >= total) return;

  const int b  = idx / per_b;
  int       r  = idx - b * per_b;
  const int h  = r / S_split;
  const int s2 = r - h * S_split;

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
  PYMDP_LAUNCH_1D(Bn * H_kk * S_split, tmp_lin_per_h_kernel<<<pymdp_launch_blocks, kBlockSize, 0, stream>>>(
                                           q01_outer, linear, Bn, K_keep, H_kk, S_split, tmp_lin));
}

// ---------------- rank-2 cuBLAS finish ----------------
//
// Reads tmp_qo[Bn, O, H_kk] from cublasSgemmStridedBatched and folds in
// entropy + linear (+ pA novelty against tmp_wa when USE_PA). One thread
// per (b, h_kk).
template <bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank2_cublas_finish_kernel(const float* __restrict__ tmp_qo, const float* __restrict__ tmp_wa,
                                                const float* __restrict__ q01_outer, const float* __restrict__ linear,
                                                int Bn, int O, int H_kk, int K_d, int64_t b_stride,
                                                float* __restrict__ score_out) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Bn * H_kk;
  if (idx >= total) return;
  const int b = idx / H_kk;
  const int h = idx % H_kk;

  float acc_ent = 0.0f;
  float pA_acc  = 0.0f;
  if (USE_STATES || USE_PA) {
    const float* tmp_b = tmp_qo + static_cast<size_t>(b) * O * H_kk;
    const float* twa_b = USE_PA ? (tmp_wa + static_cast<size_t>(b) * O * H_kk) : nullptr;
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
    for (int k = 0; k < K_d; ++k) linear_acc += q01_b[k * H_kk + h] * lin_b[k];
  }

  write_modality_score<USE_STATES>(score_out, b, b_stride, h, acc_ent, linear_acc + pA_acc);
}

cudaError_t launch_modality_score_dedup_rank2_cublas_finish(const float* tmp_qo, const float* tmp_wa,
                                                            const float* q01_outer, const float* linear, int Bn, int O,
                                                            int H_kk, int K_d, int64_t b_stride, bool use_states,
                                                            bool use_linear, bool use_pA, float* score_out,
                                                            cudaStream_t stream) {
  const int total = Bn * H_kk;
  if (total <= 0) return cudaSuccess;
  const int blocks = launch_blocks(total);
#define LAUNCH_R2_FINISH(S, L, P)                                                                                      \
  modality_score_dedup_rank2_cublas_finish_kernel<S, L, P>                                                             \
      <<<blocks, kBlockSize, 0, stream>>>(tmp_qo, tmp_wa, q01_outer, linear, Bn, O, H_kk, K_d, b_stride, score_out)
  PYMDP_DISPATCH_BOOL3(use_states, use_linear, use_pA, LAUNCH_R2_FINISH);
#undef LAUNCH_R2_FINISH
  return cudaGetLastError();
}

// =============================================================================
// Rank-3 dedup split stage 2 (finish + entropy + linear).
//
// One thread per (b, h_keep_0, h_keep_1, h_split). For each o, sum tmp_qo
// over s_split (weighted by qs_split[h_split, s_split]) to qo[o]; entropy
// then folds in. linear_acc similarly sums tmp_lin * qs_split. When USE_PA
// is set, the same o-loop folds in pA novelty (qo[o] * sum_s tmp_wa[o, s]
// * qs_split[s]).
//
// Output: score_out[b, h_keep_0 * (H_keep_1 * H_split) + h_keep_1 *
// H_split + h_split].
// =============================================================================

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
__global__ void
modality_score_dedup_rank3_split_stage2_kernel(const float* __restrict__ tmp_qo, const float* __restrict__ tmp_lin,
                                               const float* __restrict__ tmp_wa, const float* __restrict__ qs_split,
                                               int Bn, int H_keep_0, int H_keep_1, int H_split, int S_split,
                                               int64_t b_stride, float* __restrict__ score_out) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Bn * H_keep_0 * H_keep_1 * H_split;
  if (idx >= total) return;

  const int H_12 = H_keep_1 * H_split;
  const int b    = idx / (H_keep_0 * H_12);
  int       rest = idx - b * (H_keep_0 * H_12);
  const int h_0  = rest / H_12;
  rest -= h_0 * H_12;
  const int h_1 = rest / H_split;
  const int h_2 = rest - h_1 * H_split;

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
      for (int s = 0; s < S_split; ++s) qo_o += LDG(tmp_qo_w[o * S_split + s]) * LDG(qs2_p[s]);
      if (USE_STATES) acc_ent += clamp_log(qo_o);
      if (USE_PA) {
        float wa_o = 0.0f;
        for (int s = 0; s < S_split; ++s) wa_o += LDG(tmp_wa_w[o * S_split + s]) * LDG(qs2_p[s]);
        pA_acc += qo_o * wa_o;
      }
    }
  }

  float linear_acc = 0.0f;
  if (USE_LINEAR) {
    for (int s = 0; s < S_split; ++s) linear_acc += LDG(tmp_lin_w[s]) * LDG(qs2_p[s]);
  }

  // score_out is the (t, m) slice within scores_concat (b-stride =
  // total_mod_entries). The in-modality flat index has h_keep_0 outermost,
  // h_split innermost; matches the pmi encoding in neg_efe_cuda.cc.
  const int64_t local_idx = static_cast<int64_t>((h_0 * H_keep_1 + h_1)) * H_split + h_2;
  write_modality_score<USE_STATES>(score_out, b, b_stride, local_idx, acc_ent, linear_acc + pA_acc);
}

namespace {

template <int O_TPL, bool USE_STATES, bool USE_LINEAR, bool USE_PA>
inline void launch_rank3_stage2_tpl(const float* tmp_qo, const float* tmp_lin, const float* tmp_wa,
                                    const float* qs_split, int Bn, int H_keep_0, int H_keep_1, int H_split, int S_split,
                                    int64_t b_stride, float* score_out, cudaStream_t stream) {
  const int total = Bn * H_keep_0 * H_keep_1 * H_split;
  if (total <= 0) return;
  const int blocks = launch_blocks(total);
  modality_score_dedup_rank3_split_stage2_kernel<O_TPL, USE_STATES, USE_LINEAR, USE_PA>
      <<<blocks, kBlockSize, 0, stream>>>(tmp_qo, tmp_lin, tmp_wa, qs_split, Bn, H_keep_0, H_keep_1, H_split, S_split,
                                          b_stride, score_out);
}

}  // namespace

cudaError_t launch_modality_score_dedup_rank3_stage2(const float* tmp_qo, const float* tmp_lin, const float* tmp_wa,
                                                     const float* qs_split, int Bn, int O, int H_keep_0, int H_keep_1,
                                                     int H_split, int S_split, int64_t b_stride, bool use_states,
                                                     bool use_linear, bool use_pA, float* score_out,
                                                     cudaStream_t stream) {
  // Empty-grid guard lives inside launch_rank3_stage2_tpl. Outer switch over
  // O picks the kernel's O_TPL; the inner PYMDP_DISPATCH_BOOL3 specializes
  // on (USE_STATES, USE_LINEAR, USE_PA) — yields up to 8 * 8 = 64 device
  // instantiations, each with all the runtime branches DCE'd.
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
  return cudaGetLastError();
}

// =============================================================================
// Inductive per-factor.
//
// ind_score_out[b, h_f] = sum_s qs_f[b, h_f, s] * v_full[b, qs_off_f + s]
// One thread per (b, h_f).
// =============================================================================

__global__ void inductive_per_factor_kernel(const float* __restrict__ qs_f, const float* __restrict__ v_full, int Bn,
                                            int H_f, int S_f, int qs_flat, int qs_off_f, int64_t b_stride,
                                            float* __restrict__ ind_score_out) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Bn * H_f;
  if (idx >= total) return;
  const int b = idx / H_f;
  const int h = idx % H_f;

  const float* qs_p = qs_f + (b * H_f + h) * S_f;
  const float* v_p  = v_full + b * qs_flat + qs_off_f;
  float        acc  = 0.0f;
  for (int s = 0; s < S_f; ++s) acc += LDG(qs_p[s]) * LDG(v_p[s]);
  // ind_score_out is the (t, f) slice within inductive_concat (shape
  // [Bn, total_ind_entries]); b-stride = total_ind_entries.
  ind_score_out[static_cast<int64_t>(b) * b_stride + h] = acc;
}

cudaError_t launch_inductive_per_factor(const float* qs_f, const float* v_full, int Bn, int H_f, int S_f, int qs_flat,
                                        int qs_off_f, int64_t b_stride, float* ind_score_out, cudaStream_t stream) {
  PYMDP_LAUNCH_1D(Bn * H_f, inductive_per_factor_kernel<<<pymdp_launch_blocks, kBlockSize, 0, stream>>>(
                                qs_f, v_full, Bn, H_f, S_f, qs_flat, qs_off_f, b_stride, ind_score_out));
}

// =============================================================================
// Per-factor inductive coefficient v_full.
//
// Mirrors precompute_inductive() in neg_efe_precompute.h.
// One block per (b, f); thread 0 computes the scalar reductions (argmax qs,
// best-depth m_f, path availability pa) and stashes them in shared memory;
// all threads then write the per-s outputs in parallel.
// =============================================================================

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
  const float* If  = I + b * I_per_batch + I_off_f;

  __shared__ int   s_mf;
  __shared__ float s_path_log_eps;

  if (threadIdx.x == 0) {
    int   idx  = 0;
    float best = qsf[0];
    for (int s = 1; s < Sf; ++s) {
      const float v = qsf[s];
      if (v > best) {
        best = v;
        idx  = s;
      }
    }

    int   m_f   = 0;
    float mbest = If[0 * Sf + idx];
    for (int i = 1; i < Df; ++i) {
      const float v = If[i * Sf + idx];
      if (v > mbest) {
        mbest = v;
        m_f   = i;
      }
    }
    if (m_f > 0) m_f -= 1;
    s_mf = m_f;

    float pa = 0.0f;
    for (int i = 0; i < Df; ++i) pa += If[i * Sf + idx];
    pa = fmaxf(0.0f, fminf(1.0f, pa));

    const float eps_val = eps[b * eps_stride];
    s_path_log_eps      = pa * logf(eps_val);
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
  // Block size 32 covers typical S_f ≤ 32; larger Sf falls back to the
  // strided per-thread loop in the kernel.
  v_full_kernel<<<dim3(Bn, F), 32, 0, stream>>>(qs_init, I, eps, eps_stride, Bn, F, qs_flat, I_per_batch, S, depth,
                                                qs_off, I_off, v_out);
  return cudaGetLastError();
}

// =============================================================================
// Final scatter (factor-history form).
//
// out[b, p] = sum_(t, m) scores_concat[b * total_mod + mod_off[t*M + m] +
//                                      pmi[t*M*P + m*P + p]]
//           + sum_(t, f) inductive_concat[b * total_ind + ind_off[t*F + f] +
//                                         p2h[t*F*P + f*P + p]]
//           + sum_(t, f) factor_scores  [b * total_ind + ind_off[t*F + f] +
//                                         p2h[t*F*P + f*P + p]]
// The inductive sum runs only when use_inductive is true; the factor_scores
// sum runs only when use_factor_scores is true. Both share ind_off / p2h
// because factor_scores has the same per-(t, f) layout as inductive_concat.
// One thread per (b, p).
// =============================================================================

template <bool USE_INDUCTIVE, bool USE_FACTOR_SCORES>
__global__ void
final_scatter_dedup_kernel(const float* __restrict__ scores_concat, const float* __restrict__ inductive_concat,
                           const float* __restrict__ factor_scores, const int32_t* __restrict__ policy_to_modality_idx,
                           const int32_t* __restrict__ policy_to_factor_history, const int64_t* __restrict__ mod_off,
                           const int64_t* __restrict__ ind_off, int Bn, int T, int M, int F, int P, int64_t total_mod,
                           int64_t total_ind, float* __restrict__ out) {
  int       idx   = blockIdx.x * blockDim.x + threadIdx.x;
  const int total = Bn * P;
  if (idx >= total) return;
  const int b = idx / P;
  const int p = idx % P;

  // Per-thread bases hoisted: total_mod / total_ind are int64 (per-batch
  // strides for scores_concat / inductive_concat), so b * total_mod is
  // genuinely 64-bit. Computing it once keeps the t/m/f inner loops at
  // int32 + int64 add. Same shape for the inductive / factor_scores
  // bases when their flags are set.
  const float* __restrict__ scores_b        = scores_concat + b * total_mod;
  const float* __restrict__ inductive_b     = USE_INDUCTIVE ? (inductive_concat + b * total_ind) : nullptr;
  const float* __restrict__ factor_scores_b = USE_FACTOR_SCORES ? (factor_scores + b * total_ind) : nullptr;

  // The (t * M * P + m * P + p) and (t * F * P + f * P + p) walks index
  // policy_to_{modality_idx,factor_history} in int32 — production T/M/F/P
  // products stay well under 2^31 (e.g. T=8 * M=8 * P=1728 = ~110k).
  float acc = 0.0f;
  for (int t = 0; t < T; ++t) {
    const int t_M   = t * M;
    const int t_M_P = t_M * P;
    for (int m = 0; m < M; ++m) {
      const int64_t off      = LDG(mod_off[t_M + m]);
      const int32_t flat_idx = LDG(policy_to_modality_idx[t_M_P + m * P + p]);
      acc += LDG(scores_b[off + flat_idx]);
    }
    if (USE_INDUCTIVE || USE_FACTOR_SCORES) {
      const int t_F   = t * F;
      const int t_F_P = t_F * P;
      if (USE_INDUCTIVE) {
        for (int f = 0; f < F; ++f) {
          const int64_t off = LDG(ind_off[t_F + f]);
          const int32_t h_f = LDG(policy_to_factor_history[t_F_P + f * P + p]);
          acc += LDG(inductive_b[off + h_f]);
        }
      }
      if (USE_FACTOR_SCORES) {
        for (int f = 0; f < F; ++f) {
          const int64_t off = LDG(ind_off[t_F + f]);
          const int32_t h_f = LDG(policy_to_factor_history[t_F_P + f * P + p]);
          acc += LDG(factor_scores_b[off + h_f]);
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
  const int total = Bn * P;
  if (total <= 0) return cudaSuccess;
  const int blocks = launch_blocks(total);
#define LAUNCH_FINAL_SCATTER(I, FS)                                                                                    \
  final_scatter_dedup_kernel<I, FS><<<blocks, kBlockSize, 0, stream>>>(                                                \
      scores_concat, inductive_concat, factor_scores, policy_to_modality_idx, policy_to_factor_history, mod_off,       \
      ind_off, Bn, T, M, F, P, total_mod, total_ind, out)
  PYMDP_DISPATCH_BOOL2(use_inductive, use_factor_scores, LAUNCH_FINAL_SCATTER);
#undef LAUNCH_FINAL_SCATTER
  return cudaGetLastError();
}

}  // namespace cuda_kernels
}  // namespace pymdp_ffi
