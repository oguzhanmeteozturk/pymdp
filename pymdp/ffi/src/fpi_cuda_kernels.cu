// Native CUDA FPI kernel — one block per batch element, all `num_iter`
// fixed-point iterations executed internally.
//
// Algorithm mirrors fpi.cc's run_fpi_kernel_host body (per-iter softmax →
// reset to log_prior → modality marginals → convergence check), but lifted
// into a single kernel so the entire FPI run stays on the CUDA stream
// without any host roundtrip. That eliminates the FpiCudaHost shim's
// per-call cudaStreamSynchronize, which was the structural cost behind the
// ~1 ms/call regression in rollout fixtures on sm_87.
//
// Per-modality marginal accumulation uses three independent passes over
// `ll_m` (one output factor per pass), trading a 3× re-read of the modality
// likelihood for atomic-free writes to log_q. The CPU kernel fuses these
// passes; on GPU the no-atomic structure is simpler, ll_m fits in L1 for
// production shapes (S0*S1*S2 < 1 KB), and Ampere's L1 hit-rate makes the
// re-reads ~free. If profiling shows ll re-fetch hurting we can revisit
// the fused approach with shmem-atomic writes.
//
// Block size = 128 (4 warps). Picked for state sizes ~32: each warp covers
// one factor's softmax in one shot, the multi-warp count gives enough
// parallel slots for the modality passes to keep an sm_87 SM busy.
// Larger blocks would waste threads on the small inner
// dims; smaller blocks underutilize the warp-shuffle reductions.

#include <cuda_runtime.h>
#include <cstdint>

#include "fpi_cuda_kernels.h"

namespace pymdp_ffi {
namespace fpi_cuda {

namespace {

// Block size 256 (8 warps): 128 was latency-stall-bound on sm_87 due to
// warp-pool starvation (too few warps to hide L1/shmem latency chains). Some
// K=3 passes run with predicate-off warps; those still help latency hiding.
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

// ----- Block-level reductions, broadcasting the result via shmem[0] -----
//
// `scratch` must be at least kNumWarps floats. Writes the final value to
// scratch[0] and __syncthreads before returning so all threads see it.
__device__ __forceinline__ float block_reduce_max(float v, float* scratch) {
  const int tid  = threadIdx.x;
  const int lane = tid & (kWarpSize - 1);
  const int warp = tid / kWarpSize;
  v              = warp_reduce_max(v);
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  if (warp == 0) {
    v = (lane < kNumWarps) ? scratch[lane] : -INFINITY;
    v = warp_reduce_max(v);
    if (lane == 0) scratch[0] = v;
  }
  __syncthreads();
  return scratch[0];
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
// Same three-pass stable softmax as `softmax_block`, but reductions are
// pure warp shuffles — no shared-memory scratch, no __syncthreads. Caller
// arranges one warp per softmax (used to run the F per-factor softmaxes
// concurrently across the block's kNumWarps warps).
//
// Fast path for n <= 32 (every production factor S_f): keep the exp result
// in a register and write the normalized value once, instead of write-then-
// read-then-rewrite via shmem. The generic path remains for any future
// shape with S_f > 32.
__device__ void softmax_warp(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int lane = threadIdx.x & (kWarpSize - 1);

  if (n <= kWarpSize) {
    const bool  active = lane < n;
    const float x      = active ? in[lane] : -INFINITY;
    const float m      = warp_reduce_max(x);
    const float e      = active ? __expf(x - m) : 0.0f;
    const float s      = warp_reduce_sum(e);
    if (active) out[lane] = e / s;
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
  const float s = warp_reduce_sum(my_sum);

  const float inv = 1.0f / s;
  for (int i = lane; i < n; i += kWarpSize) {
    out[i] *= inv;
  }
}

// ----- K=1 modality: log_q_d0[s] += ll_m[s] -----
__device__ void modality_K1(const float* __restrict__ ll_m, int S0, float* __restrict__ log_q_d0) {
  for (int s = threadIdx.x; s < S0; s += kBlockSize) {
    log_q_d0[s] += ll_m[s];
  }
  __syncthreads();
}

// ----- K=2 modality, two-pass with kTpO-way split across the inner reduce -----
//
// Each output state is owned by `kTpO` consecutive lanes. The lanes split
// the inner reduction range, then warp-shuffle-reduce inside the group;
// lane 0 of each group writes the final sum. Pure warp-shuffle reduce with
// zero __syncthreads inside the pass.
//
// IMPORTANT: the group_reduce_sum shuffles must execute on ALL lanes of every
// warp that touches the pass, regardless of whether the lane's output index
// is in range. If we gated the shuffle behind `if (out_id < S)` and a warp
// straddled the boundary (e.g. S=6 with kTpO=4 → out_ids 0..5 enter, 6..7
// skip), the active lanes would call __shfl_xor_sync(0xffffffff, ...) with
// inactive lanes 24..31 missing — undefined behavior that hangs on Ampere.
// Instead we iterate stripes, let out-of-range lanes contribute sum=0, and
// run the shuffle unconditionally; only the in-range lane 0 of each group
// writes back.
template <int kTpO>
__device__ void modality_K2_split(const float* __restrict__ ll_m, int S0, int S1, const float* __restrict__ q0,
                                  const float* __restrict__ q1, float* __restrict__ log_q_d0,
                                  float* __restrict__ log_q_d1) {
  const int tid    = threadIdx.x;
  const int out_id = tid / kTpO;
  const int red_id = tid & (kTpO - 1);
  const int stride = kBlockSize / kTpO;

  // No __syncthreads between Pass 1 and Pass 2: Pass 1 writes log_q_d0 and
  // Pass 2 writes log_q_d1 (distinct factor slices by FFI contract on
  // lp_offs), and neither pass reads log_q within this modality — they only
  // read q (stable from the softmax phase) and ll. The trailing barrier at
  // the end of the function covers cross-modality WAW on shared factors.

  // Inner FMA loops use 4-way accumulator splitting to break the single-
  // dependency-chain pattern (`sum_{i+1} = sum_i + ...`). A single
  // accumulator stalls on the ~4-cycle Ampere FMA latency register chain;
  // with 4 independent accumulators the compiler can issue 4 FMAs per cycle
  // pair instead of 1, a ~4x ILP improvement that directly attacks that stall.
  // Tail handling for non-multiple-of-4 strides (production shapes are all
  // multiples of 4 so the tail is empty in the hot path).

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
  __syncthreads();
}

// ----- K=3 modality, three-pass with per-pass kTpO split across the inner reduce -----
//
// Same structure as modality_K2_split, but the three passes carry their own
// threads-per-output template parameter so each can be sized to its output
// dim independently. For production shapes (S0=32, S1=24, S2=16) with the
// previous single kTpO=4, the block utilization across passes was:
//
//     Pass 1 (out=S0=32): 32 * 4 = 128  -> 100%
//     Pass 2 (out=S1=24): 24 * 4 =  96  ->  75%
//     Pass 3 (out=S2=16): 16 * 4 =  64  ->  50%  (heaviest pass — half idle)
//
// Pass 3 has both the largest inner reduction (S0*S1 = 768 vs 384/512 for
// passes 1/2) AND the worst utilization. Bumping its kTpO to 8 lifts it to
// 16 * 8 = 128 (full block) and halves per-lane inner work (96 elems vs
// 192). group_reduce_sum<8> adds one extra shuffle step (3 vs 2) — cheap
// next to the per-element FMA savings.
//
// Production callers pass <4, 4, 8>. Pass 1 stays at 4 (already optimal),
// pass 2 stays at 4 (kTpO=8 would over-subscribe: 24*8=192 > 128). Other
// shapes can override the template — the K=2 helper still uses one
// shared kTpO since both its passes have the same output-dim budget
// (kTpO=4 fits S=[16..32] models comfortably).
template <int kTpO_p1, int kTpO_p2, int kTpO_p3>
__device__ void modality_K3_split(const float* __restrict__ ll_m, int S0, int S1, int S2,
                                  const float* __restrict__ q0, const float* __restrict__ q1,
                                  const float* __restrict__ q2, float* __restrict__ log_q_d0,
                                  float* __restrict__ log_q_d1, float* __restrict__ log_q_d2) {
  const int tid = threadIdx.x;
  const int S12 = S1 * S2;

  // See header comment on modality_K2_split for why the shuffle must run on
  // all lanes regardless of out-of-range out_id, and for why intra-pass
  // __syncthreads barriers can be dropped (each pass writes a distinct
  // factor slice; the trailing sync at function exit handles cross-modality
  // WAW on shared factors).
  //
  // Inner reductions are split across the kTpO lanes on the OUTER contracted
  // dim (not on the flat S_a * S_b range) so we avoid the `i / S_b`,
  // `i - s_a * S_b` divisions that nvcc otherwise emits as 32-bit divs in
  // the hot inner loop. The outer split is slightly less balanced when the
  // outer dim is < kTpO (e.g. S0=4 with kTpO=4 → only one row per lane), but
  // for production shapes the contracted outer dim ≥ 8 and the divide-free
  // form wins on Maxwell+Ampere where integer div is a multi-cycle uop.

  // Inner FMA loops use 4-way accumulator splitting per pass to break the
  // single-dependency-chain pattern. See modality_K2_split's header for the
  // ILP rationale (~4x compiler issue width vs single accumulator). The
  // accumulators are kept across the per-lane outer loop so the dependency
  // chain depth is total_inner_work / 4 instead of total_inner_work, where
  // total_inner_work for production K=3 is e.g. (S1/kTpO_p1) * S2 = 6*16 =
  // 96 FMAs serialized through one accumulator.

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
  __syncthreads();
}

// ----- Main FPI kernel — one block per batch element -----
//
// Shared-memory layout (offsets in floats, all sized total_S):
//   [0, total_S)              log_q
//   [total_S, 2*total_S)      q
//   [2*total_S, 3*total_S)    log_q_prev (convergence snapshot)
//   [3*total_S, 3*total_S+kNumWarps)  reduce scratch
//
// Total dynamic shmem = (3 * total_S + kNumWarps) floats ≈ 4 * total_S * 4 B.
// For production shapes (total_S ~ 100) that's ~1.6 KB per block — trivial,
// and one block per SM is the right occupancy for this small-batch regime.
//
// The log_q -> log_q_prev snapshot is folded into the per-iter softmax pass
// (see the softmax dispatch loop below): each warp that softmaxes a factor
// also writes that factor's log_q slice to log_q_prev while log_q is hot in
// shmem from the softmax load. The block-wide barrier closing the softmax
// chunk covers both writes, so the convergence check needs no extra
// __syncthreads — the only convergence-side cost in steady state is the
// block_reduce_max at iter end.
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
                           const ModalityDispatchGpu* __restrict__ mods)         // [M]
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

  // Threads-per-output for the K=2/K=3 split-reduce passes. kTpO=4 picks the
  // sweet spot for production shapes (S_f ≤ 32): up to 32 outputs handled in
  // parallel by 32*4=128 threads, vs the pre-split design where 32 outputs
  // ran on a single warp with 3 warps idle. Larger kTpO would help shapes
  // with very small S_f but would idle lanes when S_f ≥ 32. Templated so the
  // compiler unrolls the group-reduce shuffle chain.
  constexpr int kTpO = 4;

  const int warp_id = threadIdx.x / kWarpSize;

  const int lane = threadIdx.x & (kWarpSize - 1);

  for (int it = 0; it < num_iter; ++it) {
    // q[f] = softmax(log_q[f]) per factor — run F softmaxes concurrently
    // across the block's kNumWarps warps. Each warp handles one factor at a
    // time using pure warp-shuffle reductions (no __syncthreads inside).
    // When F > kNumWarps we loop in chunks of kNumWarps.
    //
    // Convergence snapshot folded in: while the warp's softmax is reading
    // log_q[off..off+n], we also write log_q[off+i] to log_q_prev[off+i] in
    // the same warp pass. This eliminates a standalone snapshot loop +
    // __syncthreads at iter start. The chunk-trailing __syncthreads below
    // covers both the softmax q-writes and the snapshot prev-writes, so no
    // extra barrier is needed. Shared memory bandwidth is plentiful at this
    // shape (~32 floats per factor) and log_q is hot in shmem from the
    // softmax load, so the fused snapshot is essentially free.
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

    // Per-modality marginal accumulation.
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
        // Per-pass kTpO: <4, 4, 8>. See modality_K3_split header — pass 3
        // (output dim S2) was at 50% block utilization with kTpO=4 for
        // production S=[32,24,16]; kTpO=8 lifts it to a full block.
        modality_K3_split<4, 4, 8>(ll_m, md.Ss[0], md.Ss[1], md.Ss[2], q + md.lp_offs[0], q + md.lp_offs[1],
                                   q + md.lp_offs[2], log_q + md.lp_offs[0], log_q + md.lp_offs[1],
                                   log_q + md.lp_offs[2]);
        break;
      default:
        // The host-side gate (can_handle_fpi_cuda_native in _fpi.py) only
        // dispatches us when all modalities are K<=3. A K>=4 modality
        // landing here is a contract violation in the dispatch — we'd
        // produce silently wrong results. CUDA has no exception path; the
        // closest we can do is a no-op that propagates as garbage output
        // and gets caught by parity tests. Defending here would cost a
        // branch in every iteration; we trust the gate.
        break;
      }
    }

    // Convergence check (skip iter 0 — log_q_prev was all-zero, delta is
    // large by construction). Mirrors the CPU kernel's early-exit. The
    // log_q_prev snapshot was filled by the softmax phase above, so this
    // costs only one block-wide max-reduce.
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
                       const ModalityDispatchGpu* mods, cudaStream_t stream) {
  if (batch <= 0 || total_S <= 0) return cudaSuccess;
  // Shared memory: log_q + q + log_q_prev (total_S each) + reduce scratch (kNumWarps floats).
  const size_t shmem_bytes = (static_cast<size_t>(3 * total_S) + kNumWarps) * sizeof(float);
  fpi_kernel<<<batch, kBlockSize, shmem_bytes, stream>>>(ll_flat, lp_flat, q_out, F, M, total_ll, total_S, num_iter, S,
                                                         lp_offsets, mods);
  return cudaGetLastError();
}

namespace {
// Empty body — measures only launch dispatch + driver overhead. Same
// grid/block/shmem footprint as fpi_kernel so the cuLaunchKernel parameter
// path and driver bookkeeping match the real launch byte-for-byte.
__global__ void fpi_noop_kernel() {}

// fpi_kernel_smallmeta — variant of fpi_kernel that takes the per-call
// dispatch table (S, lp_offsets, mods) by value as kernel arguments,
// served from the sm_70+ constant-memory parameter bank. Reads
// to `meta.S[f]` / `meta.lp_offsets[f]` / `meta.mods[m]` lower to LDC
// (broadcast to all 32 lanes in one cycle) instead of the pointer-fed
// kernel's LDG (10-20 cycles even on L1 hit). Used when F <= kMaxFSmallMeta
// and M <= kMaxMSmallMeta — the host gate in fpi.cc enforces this.
//
// IMPORTANT — keep in sync with fpi_kernel above. The body below is a
// verbatim copy with three substitutions:
//   * S[f]          -> meta.S[f]
//   * lp_offsets[f] -> meta.lp_offsets[f]
//   * mods[m]       -> meta.mods[m]
// The duplication is deliberate: refactoring the bodies into a shared
// templated impl risks shifting the existing fpi_kernel's SASS, and
// per feedback_fpi_cuda_no_op_changes that kernel is SASS-fragile.
// If you change algorithm in one, change it in the other; the
// scripts/sass_diff.sh harness will flag drift.
__global__ void fpi_kernel_smallmeta(const float* __restrict__ ll_flat,    // [batch, total_ll]
                                     const float* __restrict__ lp_flat,    // [batch, total_S]
                                     float* __restrict__       q_out,      // [batch, total_S]
                                     int                       F,
                                     int                       M,
                                     int                       total_ll,
                                     int                       total_S,
                                     int                       num_iter,
                                     FpiSmallMeta              meta)        // by-value, in cmem param bank
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
        // Per-pass kTpO: <4, 4, 8>. See modality_K3_split header — pass 3
        // (output dim S2) was at 50% block utilization with kTpO=4 for
        // production S=[32,24,16]; kTpO=8 lifts it to a full block.
        modality_K3_split<4, 4, 8>(ll_m, md.Ss[0], md.Ss[1], md.Ss[2], q + md.lp_offs[0], q + md.lp_offs[1],
                                   q + md.lp_offs[2], log_q + md.lp_offs[0], log_q + md.lp_offs[1],
                                   log_q + md.lp_offs[2]);
        break;
      default:
        break;
      }
    }

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
                                 cudaStream_t stream) {
  if (batch <= 0 || total_S <= 0) return cudaSuccess;
  const size_t shmem_bytes = (static_cast<size_t>(3 * total_S) + kNumWarps) * sizeof(float);
  fpi_kernel_smallmeta<<<batch, kBlockSize, shmem_bytes, stream>>>(ll_flat, lp_flat, q_out, F, M, total_ll, total_S,
                                                                   num_iter, meta);
  return cudaGetLastError();
}

}  // namespace fpi_cuda
}  // namespace pymdp_ffi
