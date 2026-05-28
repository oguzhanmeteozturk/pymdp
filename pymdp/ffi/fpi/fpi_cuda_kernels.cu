// Native CUDA FPI kernel — one block per batch element, all `num_iter`
// fixed-point iterations executed internally.
//
// Algorithm mirrors fpi_cpu_runner.cc's run_fpi_kernel_host body (per-iter
// softmax → reset to log_prior → modality marginals → convergence check),
// lifted into a single kernel so the whole run stays on the CUDA stream with
// no host roundtrip.
//
// Per-modality marginal accumulation uses K independent passes over `ll_m`
// (one output factor per pass), trading a re-read of the modality likelihood
// for atomic-free writes to log_q. ll_m fits in L1 for production shapes
// (S0*S1*S2 < 1 KB) so the re-reads stay cheap.

#include <cuda_runtime.h>
#include <cstdint>

#include "fpi/fpi_cuda_kernels.h"

namespace pymdp_ffi {
namespace fpi_cuda {

namespace {

// Block size 256 (8 warps). K=3 / K=2 split-reduce widths (kTpO) target full
// block utilization at this size — see modality_K3_split's header.
constexpr int kBlockSize = 256;
constexpr int kWarpSize  = 32;
constexpr int kNumWarps  = kBlockSize / kWarpSize;  // 8

// ----- Warp-level reductions (full 32-lane mask) -----
__device__ __forceinline__ float warp_reduce_max(float v) {
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 2));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 1));
  return v;
}

__device__ __forceinline__ float warp_reduce_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

// ----- Block-level reductions via single-sync redundant-warp pattern -----
//
// `scratch` must be at least kNumWarps floats. Returns the block-wide max to
// every thread. Every warp redundantly reads scratch[0..kNumWarps) and
// warp-shuffle-reduces, so no broadcast sync is needed. Lanes [kNumWarps, 32)
// seed from -INFINITY so warp_reduce_max returns the correct max-over-warps.
__device__ __forceinline__ float block_reduce_max(float v, float* scratch) {
  const int tid  = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  v              = warp_reduce_max(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  // All warps redundantly compute the cross-warp max from scratch.
  v = (lane < kNumWarps) ? scratch[lane] : -INFINITY;
  v = warp_reduce_max(v);
  return v;
}

// ----- Partial warp-shuffle sum over a group of kTpO lanes (power of 2) -----
//
// Each output state in the modality split passes is owned by a kTpO-sized
// lane group. After per-lane partial-sum accumulation across the inner
// reduction range, we shuffle-reduce inside the group. kTpO must divide 32
// so the group lives within a warp (cross-warp shuffles are not supported).
template <int kTpO>
__device__ __forceinline__ float group_reduce_sum(float v) {
  static_assert(kTpO == 1 || kTpO == 2 || kTpO == 4 || kTpO == 8 || kTpO == 16 || kTpO == 32,
                "kTpO must be a power of 2 between 1 and 32");
#pragma unroll
  for (int off = kTpO / 2; off > 0; off >>= 1) {
    v += __shfl_xor_sync(0xffffffffu, v, off);
  }
  return v;
}

// ----- Single-warp softmax over `n` floats -----
//
// Three-pass stable softmax with pure warp-shuffle reductions; one warp per
// softmax. Fast path for n <= 32 keeps the exp result in a register. A
// degenerate sum (all-`-inf` input -> s is 0 or NaN) falls back to uniform
// 1/n, mirroring softmax_inplace.
__device__ void softmax_warp(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int lane = threadIdx.x & (kWarpSize - 1);

  if (n <= kWarpSize) {
    const bool  active = lane < n;
    const float x      = active ? in[lane] : -INFINITY;
    const float m      = warp_reduce_max(x);
    const float e      = active ? __expf(x - m) : 0.0f;
    const float s      = warp_reduce_sum(e);
    const bool  ok     = s > 0.0f && isfinite(s);
    if (active) out[lane] = ok ? e / s : 1.0f / n;
    return;
  }

  float my_max = -INFINITY;
  for (int i = lane; i < n; i += kWarpSize) {
    my_max = fmaxf(my_max, in[i]);
  }
  const float m = warp_reduce_max(my_max);

  float my_sum = 0.0f;
  for (int i = lane; i < n; i += kWarpSize) {
    const float e = __expf(in[i] - m);
    out[i]        = e;
    my_sum += e;
  }
  const float s   = warp_reduce_sum(my_sum);
  const bool  ok  = s > 0.0f && isfinite(s);
  const float inv = ok ? 1.0f / s : 0.0f;
  for (int i = lane; i < n; i += kWarpSize) {
    out[i] = ok ? out[i] * inv : 1.0f / n;
  }
}

// ----- K=1 modality: log_q_d0[s] += ll_m[s] -----
//
// No trailing __syncthreads here — the caller decides whether the cross-
// modality WAW barrier is needed based on sync_mask. See the modality loop
// in fpi_kernel / fpi_kernel_smallmeta.
__device__ void modality_K1(const float* __restrict__ ll_m, int S0, float* __restrict__ log_q_d0) {
  for (int s = threadIdx.x; s < S0; s += kBlockSize) {
    log_q_d0[s] += ll_m[s];
  }
}

// ----- K=2 modality, two-pass with kTpO-way split across the inner reduce -----
//
// Each output state is owned by `kTpO` consecutive lanes. The lanes split
// the inner reduction range, then warp-shuffle-reduce inside the group;
// lane 0 of each group writes the final sum. Pure warp-shuffle reduce with
// zero __syncthreads inside the pass.
//
// IMPORTANT: group_reduce_sum's shuffles must run on ALL lanes regardless of
// whether the lane's output index is in range. Gating the shuffle behind
// `if (out_id < S)` when a warp straddles the boundary calls
// __shfl_xor_sync(0xffffffff, ...) with lanes missing — UB that hangs on
// Ampere. Instead, iterate stripes, let out-of-range lanes contribute sum=0,
// run the shuffle unconditionally, and only the in-range lane 0 writes back.
template <int kTpO>
__device__ void modality_K2_split(const float* __restrict__ ll_m, int S0, int S1, const float* __restrict__ q0,
                                  const float* __restrict__ q1, float* __restrict__ log_q_d0,
                                  float* __restrict__ log_q_d1) {
  const int tid    = threadIdx.x;
  const int out_id = tid / kTpO;
  const int red_id = tid & (kTpO - 1);
  const int stride = kBlockSize / kTpO;

  // No intra-modality __syncthreads: the two passes write distinct factor
  // slices (lp_offs) and read only q (stable from softmax) and ll. The
  // function-exit barrier covers cross-modality WAW on shared factors.
  //
  // Inner FMA loops use 4-way accumulator splitting for ILP; the scalar tail
  // handles non-multiple-of-4 strides (empty for production shapes).

  // Pass 1: log_q_d0[s0] += sum_{s1} ll[s0,s1] * q1[s1]
  for (int s0_base = 0; s0_base < S0; s0_base += stride) {
    const int s0  = s0_base + out_id;
    float     sum = 0.0f;
    if (s0 < S0) {
      const float* row    = ll_m + s0 * S1;
      const int    step   = 4 * kTpO;  // four lanes' worth of inner stride
      float        a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      int          s1 = red_id;
      for (; s1 + 3 * kTpO < S1; s1 += step) {
        a0 += row[s1 + 0 * kTpO] * q1[s1 + 0 * kTpO];
        a1 += row[s1 + 1 * kTpO] * q1[s1 + 1 * kTpO];
        a2 += row[s1 + 2 * kTpO] * q1[s1 + 2 * kTpO];
        a3 += row[s1 + 3 * kTpO] * q1[s1 + 3 * kTpO];
      }
      for (; s1 < S1; s1 += kTpO) {
        a0 += row[s1] * q1[s1];
      }
      sum = (a0 + a1) + (a2 + a3);
    }
    sum = group_reduce_sum<kTpO>(sum);
    if (s0 < S0 && red_id == 0) log_q_d0[s0] += sum;
  }
  // Pass 2: log_q_d1[s1] += sum_{s0} ll[s0,s1] * q0[s0]
  for (int s1_base = 0; s1_base < S1; s1_base += stride) {
    const int s1  = s1_base + out_id;
    float     sum = 0.0f;
    if (s1 < S1) {
      const int step = 4 * kTpO;
      float     a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
      int       s0 = red_id;
      for (; s0 + 3 * kTpO < S0; s0 += step) {
        a0 += ll_m[(s0 + 0 * kTpO) * S1 + s1] * q0[s0 + 0 * kTpO];
        a1 += ll_m[(s0 + 1 * kTpO) * S1 + s1] * q0[s0 + 1 * kTpO];
        a2 += ll_m[(s0 + 2 * kTpO) * S1 + s1] * q0[s0 + 2 * kTpO];
        a3 += ll_m[(s0 + 3 * kTpO) * S1 + s1] * q0[s0 + 3 * kTpO];
      }
      for (; s0 < S0; s0 += kTpO) {
        a0 += ll_m[s0 * S1 + s1] * q0[s0];
      }
      sum = (a0 + a1) + (a2 + a3);
    }
    sum = group_reduce_sum<kTpO>(sum);
    if (s1 < S1 && red_id == 0) log_q_d1[s1] += sum;
  }
  // No trailing __syncthreads — caller gates the cross-modality WAW barrier
  // via sync_mask. See the modality loop in fpi_kernel / fpi_kernel_smallmeta.
}

// ----- K=3 modality, three-pass with per-pass kTpO split across the inner reduce -----
//
// Same structure as modality_K2_split, but each of the three passes carries
// its own threads-per-output template parameter so it can be sized to its
// output dim independently. The shipping callsite uses <8, 8, 8>, tuned
// empirically on Orin sm_87 for production shapes S=[32,24,16].
template <int kTpO_p1, int kTpO_p2, int kTpO_p3>
__device__ void modality_K3_split(const float* __restrict__ ll_m, int S0, int S1, int S2,
                                  const float* __restrict__ q0, const float* __restrict__ q1,
                                  const float* __restrict__ q2, float* __restrict__ log_q_d0,
                                  float* __restrict__ log_q_d1, float* __restrict__ log_q_d2) {
  const int tid = threadIdx.x;
  const int S12 = S1 * S2;

  // See modality_K2_split's header for why the shuffle must run on all lanes
  // and why intra-pass __syncthreads can be dropped. Inner reductions split
  // across the kTpO lanes on the OUTER contracted dim (not the flat S_a*S_b
  // range) to avoid the per-iter integer divisions nvcc would otherwise emit.
  // 4-way accumulator splitting per pass for ILP.

  // Pass 1: log_q_d0[s0] += sum_{s1, s2} ll[s0,s1,s2] * q1[s1] * q2[s2]
  {
    const int out_id = tid / kTpO_p1;
    const int red_id = tid & (kTpO_p1 - 1);
    const int stride = kBlockSize / kTpO_p1;
    for (int s0_base = 0; s0_base < S0; s0_base += stride) {
      const int s0  = s0_base + out_id;
      float     sum = 0.0f;
      if (s0 < S0) {
        const float* base = ll_m + s0 * S12;
        float        a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s1 = red_id; s1 < S1; s1 += kTpO_p1) {
          const float  q1v = q1[s1];
          const float* row = base + s1 * S2;
          int          s2  = 0;
          for (; s2 + 4 <= S2; s2 += 4) {
            a0 += row[s2 + 0] * q1v * q2[s2 + 0];
            a1 += row[s2 + 1] * q1v * q2[s2 + 1];
            a2 += row[s2 + 2] * q1v * q2[s2 + 2];
            a3 += row[s2 + 3] * q1v * q2[s2 + 3];
          }
          for (; s2 < S2; ++s2) {
            a0 += row[s2] * q1v * q2[s2];
          }
        }
        sum = (a0 + a1) + (a2 + a3);
      }
      sum = group_reduce_sum<kTpO_p1>(sum);
      if (s0 < S0 && red_id == 0) log_q_d0[s0] += sum;
    }
  }
  // Pass 2: log_q_d1[s1] += sum_{s0, s2} ll[s0,s1,s2] * q0[s0] * q2[s2]
  {
    const int out_id = tid / kTpO_p2;
    const int red_id = tid & (kTpO_p2 - 1);
    const int stride = kBlockSize / kTpO_p2;
    for (int s1_base = 0; s1_base < S1; s1_base += stride) {
      const int s1  = s1_base + out_id;
      float     sum = 0.0f;
      if (s1 < S1) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = red_id; s0 < S0; s0 += kTpO_p2) {
          const float  q0v = q0[s0];
          const float* row = ll_m + s0 * S12 + s1 * S2;
          int          s2  = 0;
          for (; s2 + 4 <= S2; s2 += 4) {
            a0 += row[s2 + 0] * q0v * q2[s2 + 0];
            a1 += row[s2 + 1] * q0v * q2[s2 + 1];
            a2 += row[s2 + 2] * q0v * q2[s2 + 2];
            a3 += row[s2 + 3] * q0v * q2[s2 + 3];
          }
          for (; s2 < S2; ++s2) {
            a0 += row[s2] * q0v * q2[s2];
          }
        }
        sum = (a0 + a1) + (a2 + a3);
      }
      sum = group_reduce_sum<kTpO_p2>(sum);
      if (s1 < S1 && red_id == 0) log_q_d1[s1] += sum;
    }
  }
  // Pass 3: log_q_d2[s2] += sum_{s0, s1} ll[s0,s1,s2] * q0[s0] * q1[s1]
  {
    const int out_id = tid / kTpO_p3;
    const int red_id = tid & (kTpO_p3 - 1);
    const int stride = kBlockSize / kTpO_p3;
    for (int s2_base = 0; s2_base < S2; s2_base += stride) {
      const int s2  = s2_base + out_id;
      float     sum = 0.0f;
      if (s2 < S2) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        for (int s0 = red_id; s0 < S0; s0 += kTpO_p3) {
          const float q0v  = q0[s0];
          const int   base = s0 * S12 + s2;  // hoisted; stride S2 across s1
          int         s1   = 0;
          for (; s1 + 4 <= S1; s1 += 4) {
            a0 += ll_m[base + (s1 + 0) * S2] * q0v * q1[s1 + 0];
            a1 += ll_m[base + (s1 + 1) * S2] * q0v * q1[s1 + 1];
            a2 += ll_m[base + (s1 + 2) * S2] * q0v * q1[s1 + 2];
            a3 += ll_m[base + (s1 + 3) * S2] * q0v * q1[s1 + 3];
          }
          for (; s1 < S1; ++s1) {
            a0 += ll_m[base + s1 * S2] * q0v * q1[s1];
          }
        }
        sum = (a0 + a1) + (a2 + a3);
      }
      sum = group_reduce_sum<kTpO_p3>(sum);
      if (s2 < S2 && red_id == 0) log_q_d2[s2] += sum;
    }
  }
  // No trailing __syncthreads — caller gates the cross-modality WAW barrier
  // via sync_mask. See the modality loop in fpi_kernel / fpi_kernel_smallmeta.
}

// ----- Main FPI kernel — one block per batch element -----
//
// Shared-memory layout (offsets in floats, all sized total_S):
//   [0, total_S)              log_q
//   [total_S, 2*total_S)      q
//   [2*total_S, 3*total_S)    log_q_prev (convergence snapshot)
//   [3*total_S, 3*total_S+kNumWarps)  reduce scratch
//
// Total dynamic shmem = (3 * total_S + kNumWarps) floats. The log_q ->
// log_q_prev convergence snapshot is folded into the per-iter softmax pass
// (see the softmax dispatch loop below), so the convergence check costs only
// the block_reduce_max at iter end.
__global__ void fpi_kernel(const float* __restrict__ ll_flat,    // [batch, total_ll]
                           const float* __restrict__ lp_flat,    // [batch, total_S]
                           float* __restrict__       q_out,      // [batch, total_S]
                           int                       F,
                           int                       M,
                           int                       total_ll,
                           int                       total_S,
                           int                       num_iter,
                           const int32_t* __restrict__             S,            // [F]
                           const int32_t* __restrict__             lp_offsets,   // [F]
                           const ModalityDispatchGpu* __restrict__ mods,         // [M]
                           uint32_t                                sync_mask)
{
  const int    b     = blockIdx.x;
  const float* ll_b  = ll_flat + static_cast<size_t>(b) * total_ll;
  const float* lp_b  = lp_flat + static_cast<size_t>(b) * total_S;
  float*       q_out_b = q_out + static_cast<size_t>(b) * total_S;

  extern __shared__ float shmem[];
  float* log_q      = shmem;
  float* q          = log_q + total_S;
  float* log_q_prev = q + total_S;
  float* scratch    = log_q_prev + total_S;  // size kNumWarps

  // Init log_q to zero — matches the CPU kernel's std::memset.
  for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
    log_q[i] = 0.0f;
  }
  __syncthreads();

  // Threads-per-output for the K=2 split-reduce passes (K=3 overrides per
  // pass at its callsite). kTpO=4 suits production shapes (S_f ≤ 32).
  constexpr int kTpO = 4;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane    = threadIdx.x & (kWarpSize - 1);

  for (int it = 0; it < num_iter; ++it) {
    // q[f] = softmax(log_q[f]) per factor — F softmaxes run concurrently
    // across the kNumWarps warps (one factor per warp, chunked when
    // F > kNumWarps). The log_q -> log_q_prev convergence snapshot is folded
    // in here, covered by the chunk-trailing __syncthreads.
    //
    // Do not hoist the trailing __syncthreads out of this loop — nvcc codegen
    // is sensitive here and the rearrangement regresses fpi_large.
    for (int f0 = 0; f0 < F; f0 += kNumWarps) {
      const int f = f0 + warp_id;
      if (f < F) {
        const int n   = S[f];
        const int off = lp_offsets[f];
        softmax_warp(log_q + off, q + off, n);
        // Snapshot this factor's log_q into log_q_prev. Same warp, same
        // factor, runs alongside softmax_warp's q-writes.
        for (int i = lane; i < n; i += kWarpSize) {
          log_q_prev[off + i] = log_q[off + i];
        }
      }
      __syncthreads();
    }

    // Reset accumulator: log_q = lp_b.
    for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
      log_q[i] = lp_b[i];
    }
    __syncthreads();

    // Per-modality marginal accumulation. The modality helpers emit no
    // trailing __syncthreads; sync_mask gates it here so non-overlapping
    // modalities run back-to-back. The unconditional post-loop sync covers
    // the final write, so bit (M-1) of sync_mask is always 0.
    // SASS-fragile: if you change this loop's shape, mirror the exact change
    // in fpi_kernel_smallmeta below and bench fpi_large first.
    for (int m = 0; m < M; ++m) {
      const ModalityDispatchGpu md   = mods[m];
      const float*              ll_m = ll_b + md.ll_off;
      switch (md.K) {
      case 1:
        modality_K1(ll_m, md.Ss[0], log_q + md.lp_offs[0]);
        break;
      case 2:
        modality_K2_split<kTpO>(ll_m, md.Ss[0], md.Ss[1], q + md.lp_offs[0], q + md.lp_offs[1],
                                log_q + md.lp_offs[0], log_q + md.lp_offs[1]);
        break;
      case 3:
        // Per-pass kTpO: <8, 8, 8> — tuned empirically on Orin sm_87. See
        // modality_K3_split header.
        modality_K3_split<8, 8, 8>(ll_m, md.Ss[0], md.Ss[1], md.Ss[2], q + md.lp_offs[0], q + md.lp_offs[1],
                                   q + md.lp_offs[2], log_q + md.lp_offs[0], log_q + md.lp_offs[1],
                                   log_q + md.lp_offs[2]);
        break;
      default:
        // K>=4 never reaches here — the host gate (can_handle_fpi_cuda_native
        // in _fpi.py) only dispatches K<=3, and the cache build validates it.
        break;
      }
      if ((sync_mask >> m) & 1u) __syncthreads();
    }
    __syncthreads();  // WAW + RAW barrier for convergence read / next softmax

    // Convergence check (skip iter 0 — log_q_prev was all-zero, delta is
    // large by construction). Mirrors the CPU kernel's early-exit.
    if (it > 0) {
      float my_max_diff = 0.0f;
      for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
        my_max_diff = fmaxf(my_max_diff, fabsf(log_q[i] - log_q_prev[i]));
      }
      const float max_diff = block_reduce_max(my_max_diff, scratch);
      if (max_diff < 1e-5f) break;
    }
  }

  // Final softmax to q_out — same concurrent-warp pattern as the per-iter
  // softmax phase.
  for (int f0 = 0; f0 < F; f0 += kNumWarps) {
    const int f = f0 + warp_id;
    if (f < F) {
      const int n   = S[f];
      const int off = lp_offsets[f];
      softmax_warp(log_q + off, q_out_b + off, n);
    }
    __syncthreads();
  }
}

}  // namespace

cudaError_t launch_fpi(const float* ll_flat, const float* lp_flat, float* q_out, int batch, int F, int M, int total_ll,
                       int total_S, int num_iter, const int32_t* S, const int32_t* lp_offsets,
                       const ModalityDispatchGpu* mods, uint32_t sync_mask, cudaStream_t stream) {
  if (batch <= 0 || total_S <= 0) return cudaSuccess;
  // Shared memory: log_q + q + log_q_prev (total_S each) + reduce scratch (kNumWarps floats).
  const size_t shmem_bytes = (static_cast<size_t>(3 * total_S) + kNumWarps) * sizeof(float);
  fpi_kernel<<<batch, kBlockSize, shmem_bytes, stream>>>(ll_flat, lp_flat, q_out, F, M, total_ll, total_S, num_iter, S,
                                                         lp_offsets, mods, sync_mask);
  return cudaGetLastError();
}

namespace {
// Empty body — measures only launch dispatch + driver overhead. Same
// grid/block/shmem footprint as fpi_kernel so the cuLaunchKernel parameter
// path and driver bookkeeping match the real launch byte-for-byte.
__global__ void fpi_noop_kernel() {}

// fpi_kernel_smallmeta — variant of fpi_kernel that takes the dispatch table
// (S, lp_offsets, mods) by value, served from the constant-memory parameter
// bank: meta.* reads lower to LDC (one-cycle broadcast) instead of the
// pointer-fed kernel's LDG. Used when F <= kMaxFSmallMeta and
// M <= kMaxMSmallMeta — the host gate in fpi_cuda_runtime.cc enforces this.
//
// IMPORTANT — keep in sync with fpi_kernel above. The body is a verbatim copy
// with S[f]/lp_offsets[f]/mods[m] replaced by meta.S[f]/meta.lp_offsets[f]/
// meta.mods[m]. The duplication is deliberate: a shared templated impl risks
// shifting fpi_kernel's SASS, which is fragile. scripts/sass_diff.sh flags drift.
__global__ void fpi_kernel_smallmeta(const float* __restrict__ ll_flat,    // [batch, total_ll]
                                     const float* __restrict__ lp_flat,    // [batch, total_S]
                                     float* __restrict__       q_out,      // [batch, total_S]
                                     int                       F,
                                     int                       M,
                                     int                       total_ll,
                                     int                       total_S,
                                     int                       num_iter,
                                     FpiSmallMeta              meta,        // by-value, in cmem param bank
                                     uint32_t                  sync_mask)
{
  const int    b     = blockIdx.x;
  const float* ll_b  = ll_flat + static_cast<size_t>(b) * total_ll;
  const float* lp_b  = lp_flat + static_cast<size_t>(b) * total_S;
  float*       q_out_b = q_out + static_cast<size_t>(b) * total_S;

  extern __shared__ float shmem[];
  float* log_q      = shmem;
  float* q          = log_q + total_S;
  float* log_q_prev = q + total_S;
  float* scratch    = log_q_prev + total_S;  // size kNumWarps

  for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
    log_q[i] = 0.0f;
  }
  __syncthreads();

  constexpr int kTpO = 4;

  const int warp_id = threadIdx.x / kWarpSize;
  const int lane    = threadIdx.x & (kWarpSize - 1);

  for (int it = 0; it < num_iter; ++it) {
    for (int f0 = 0; f0 < F; f0 += kNumWarps) {
      const int f = f0 + warp_id;
      if (f < F) {
        const int n   = meta.S[f];
        const int off = meta.lp_offsets[f];
        softmax_warp(log_q + off, q + off, n);
        for (int i = lane; i < n; i += kWarpSize) {
          log_q_prev[off + i] = log_q[off + i];
        }
      }
      __syncthreads();
    }

    for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
      log_q[i] = lp_b[i];
    }
    __syncthreads();

    // Mirror of the fpi_kernel modality loop above — sync_mask gates the
    // per-modality WAW barrier, post-loop unconditional barrier covers the
    // final write. Keep these two loops byte-for-byte identical (modulo the
    // mods[m] vs meta.mods[m] substitution) per the in-sync contract.
    for (int m = 0; m < M; ++m) {
      const ModalityDispatchGpu md   = meta.mods[m];
      const float*              ll_m = ll_b + md.ll_off;
      switch (md.K) {
      case 1:
        modality_K1(ll_m, md.Ss[0], log_q + md.lp_offs[0]);
        break;
      case 2:
        modality_K2_split<kTpO>(ll_m, md.Ss[0], md.Ss[1], q + md.lp_offs[0], q + md.lp_offs[1],
                                log_q + md.lp_offs[0], log_q + md.lp_offs[1]);
        break;
      case 3:
        // Per-pass kTpO: <8, 8, 8> — tuned empirically on Orin sm_87. See
        // modality_K3_split header.
        modality_K3_split<8, 8, 8>(ll_m, md.Ss[0], md.Ss[1], md.Ss[2], q + md.lp_offs[0], q + md.lp_offs[1],
                                   q + md.lp_offs[2], log_q + md.lp_offs[0], log_q + md.lp_offs[1],
                                   log_q + md.lp_offs[2]);
        break;
      default:
        break;
      }
      if ((sync_mask >> m) & 1u) __syncthreads();
    }
    __syncthreads();  // WAW + RAW barrier for convergence read / next softmax

    if (it > 0) {
      float my_max_diff = 0.0f;
      for (int i = threadIdx.x; i < total_S; i += kBlockSize) {
        my_max_diff = fmaxf(my_max_diff, fabsf(log_q[i] - log_q_prev[i]));
      }
      const float max_diff = block_reduce_max(my_max_diff, scratch);
      if (max_diff < 1e-5f) break;
    }
  }

  for (int f0 = 0; f0 < F; f0 += kNumWarps) {
    const int f = f0 + warp_id;
    if (f < F) {
      const int n   = meta.S[f];
      const int off = meta.lp_offsets[f];
      softmax_warp(log_q + off, q_out_b + off, n);
    }
    __syncthreads();
  }
}
}  // namespace

cudaError_t launch_fpi_noop(int batch, int total_S, cudaStream_t stream) {
  if (batch <= 0 || total_S <= 0) return cudaSuccess;
  const size_t shmem_bytes = (static_cast<size_t>(3 * total_S) + kNumWarps) * sizeof(float);
  fpi_noop_kernel<<<batch, kBlockSize, shmem_bytes, stream>>>();
  return cudaGetLastError();
}

cudaError_t launch_fpi_smallmeta(const float* ll_flat, const float* lp_flat, float* q_out, int batch, int F, int M,
                                 int total_ll, int total_S, int num_iter, const FpiSmallMeta& meta,
                                 uint32_t sync_mask, cudaStream_t stream) {
  if (batch <= 0 || total_S <= 0) return cudaSuccess;
  const size_t shmem_bytes = (static_cast<size_t>(3 * total_S) + kNumWarps) * sizeof(float);
  fpi_kernel_smallmeta<<<batch, kBlockSize, shmem_bytes, stream>>>(ll_flat, lp_flat, q_out, F, M, total_ll, total_S,
                                                                   num_iter, meta, sync_mask);
  return cudaGetLastError();
}

}  // namespace fpi_cuda
}  // namespace pymdp_ffi
