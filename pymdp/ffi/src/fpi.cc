// Generalized fused fixed-point iteration (FPI) kernel for pymdp.
//
// Replaces the JAX `lax.scan` body in `pymdp.algos.run_factorized_fpi` with one
// C++ call that runs all `num_iter` iterations internally. Supports arbitrary
// factor/modal counts and state sizes, with modality dep rank in [1, 8];
// higher ranks fall back to JAX via can_handle_fpi.
//
// Per-iteration body (matches algos.py:run_factorized_fpi scan_fn exactly):
//   q[f]      = softmax(log_q[f])
//   log_q[f]  = log_prior[f] + sum_m sum_{l : A_deps[m][l] == f} marg[m, l]
//   where marg[m, l] = factor_dot(ll_m, [q[A_deps[m][j]] for j != l],
//                                 keep_dims=(l,))
//
// K=3 modalities use shared prefixes to avoid duplicate contractions; the
// modality_K3 helper documents the concrete t01/t12 schedule.
//
// K>=4 modalities use a leave-one-out path (modality_Kn): for each output
// factor k, contract ll against suffix_q[k] = q_{k+1}⊗...⊗q_{K-1} (sgemv) then
// fold against prefix_q[k] = q_0⊗...⊗q_{k-1} (axpy). suffix tensors are
// precomputed back-to-back into a single buffer; prefix is extended in-place
// (reverse-order kron) per-k. Inner sgemv/axpy auto-vectorize regardless of
// runtime K so this matches the K=3 cost-per-element on a per-k basis.
//
// ABI: flat ll/prior buffers in, flat q buffer out; attrs carry state sizes,
// offsets, dependencies, and num_iter. Under vmap_method="broadcast_all" the
// kernel infers batch = lp_flat.size / total_S and loops over batch elements
// inside one XLA dispatch.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <omp.h>

#ifdef PYMDP_FFI_HAS_CUDA
#include <cuda_runtime.h>
#include "cuda_host_alias.h"
#include "fpi_cuda_kernels.h"
#include "neg_efe_cuda_memory.h"
#endif

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "error_helpers.h"
#include "fpi.h"
#include "neg_efe_layout.h"
#include "kernel_primitives.h"
#include "omp_helpers.h"

namespace ffi = ::xla::ffi;

namespace pymdp_ffi {
namespace {

constexpr const char* kFpiKernelName = "fpi_ffi";

// =============================================================================
// Per-modality inner ops
// =============================================================================

// Per-factor softmax: copies src[off..off+S[f]] into dst and replaces in-place
// with stable softmax. Same shape signature as the per-iteration q[f] update
// and the final q_out write. src/dst are distinct buffers (scratch_log_q vs
// scratch_q in fpi_one_batch, scratch_log_q vs q_out at the tail), so the
// memcpy is non-overlapping and restrict applies.
inline void softmax_per_factor(int64_t F, const int64_t* __restrict__ S, const int64_t* __restrict__ lp_offsets,
                               const float* __restrict__ src, float* __restrict__ dst) {
  for (int64_t f = 0; f < F; ++f) {
    const int64_t n   = S[f];
    const int64_t off = lp_offsets[f];
    std::memcpy(dst + off, src + off, n * sizeof(float));
    softmax_inplace(dst + off, n);
  }
}

// Per-factor softmax + convergence snapshot. Identical to softmax_per_factor
// but additionally writes src[off..off+S[f]] into prev[off..off+S[f]] — the
// convergence check's log_q_prev. Compiles to a single src-load loop that
// fans out to both dst and prev, so the snapshot piggybacks on a load that
// the softmax pass would have done anyway. Folding the snapshot here means
// fpi_one_batch can drop the explicit `std::memcpy(scratch_log_q_prev,
// scratch_log_q, total_S * sizeof(float))` from the per-iter prelude — net
// save is one full-buffer pass over scratch_log_q per iter.
//
// src/dst/prev are all distinct: src = scratch_log_q, dst = scratch_q,
// prev = scratch_log_q_prev (three distinct FpiScratch vectors).
inline void softmax_per_factor_snapshot(int64_t F, const int64_t* __restrict__ S,
                                        const int64_t* __restrict__ lp_offsets, const float* __restrict__ src,
                                        float* __restrict__ dst, float* __restrict__ prev) {
  for (int64_t f = 0; f < F; ++f) {
    const int64_t n   = S[f];
    const int64_t off = lp_offsets[f];
    // Single-pass read of src fanning out to dst and prev. Autovectorizes to
    // a NEON/AVX2 load + two stores; same memory-bandwidth footprint as the
    // bare memcpy in softmax_per_factor since src is hot for the duration.
    const float* s = src + off;
    float*       d = dst + off;
    float*       p = prev + off;
    for (int64_t i = 0; i < n; ++i) {
      const float v = s[i];
      d[i]          = v;
      p[i]          = v;
    }
    softmax_inplace(d, n);
  }
}

// K=1 modality: ll_m has shape (S0,). marg[s0] = ll_m[s0] (no contraction).
inline void modality_K1(const float* __restrict__ ll_m, int64_t S0, float* __restrict__ log_q_d0) {
  for (int64_t s = 0; s < S0; ++s) log_q_d0[s] += ll_m[s];
}

// K=2 modality (ll[s0, s1] of shape S0×S1, row-major):
//   marg0[s0] = sum_{s1} ll[s0,s1] * q1[s1]   → log_q[deps[0]]
//   marg1[s1] = sum_{s0} ll[s0,s1] * q0[s0]   → log_q[deps[1]]
//
// Fused single-pass over ll: per (s0, s1) read v = ll[s0,s1] once and feed
// both a per-row reduction (marg0) and a per-column RMW (marg1), halving ll
// bandwidth when S0*S1 straddles the L1 boundary. The inner s1 loop
// auto-vectorizes — both streams (q1 read, log_q_d1 RMW) are unit-stride
// contiguous, so the FMA for sum_d0 and the FMA for log_q_d1[s1] += qs0*v
// share v in registers.
// All five pointers reference distinct memory: ll_m is in ll_flat, q0/q1 are
// non-overlapping slices of scratch_q (distinct lp_offs guaranteed by the
// dedup check in FpiCpu), log_q_d0/log_q_d1 are non-overlapping slices of
// scratch_log_q. Restrict lets the compiler keep q1[s1] and qs0 in registers
// across the log_q_d1 RMW.
inline void modality_K2(const float* __restrict__ ll_m, int64_t S0, int64_t S1, const float* __restrict__ q0,
                        const float* __restrict__ q1, float* __restrict__ log_q_d0, float* __restrict__ log_q_d1) {
  for (int64_t s0 = 0; s0 < S0; ++s0) {
    const float  qs0    = q0[s0];
    const float* ll_row = ll_m + s0 * S1;
    float        sum_d0 = 0.0f;
    for (int64_t s1 = 0; s1 < S1; ++s1) {
      const float v = ll_row[s1];
      sum_d0 += v * q1[s1];
      log_q_d1[s1] += qs0 * v;
    }
    log_q_d0[s0] += sum_d0;
  }
}

// K=3 modality (ll[s0, s1, s2] of shape S0×S1×S2, row-major):
//   marg0[s0]  = sum_{s1, s2} ll[s0,s1,s2] * q1[s1] * q2[s2]   → log_q[deps[0]]
//   marg1[s1]  = sum_{s0, s2} ll[s0,s1,s2] * q0[s0] * q2[s2]   → log_q[deps[1]]
//   marg2[s2]  = sum_{s0, s1} ll[s0,s1,s2] * q0[s0] * q1[s1]   → log_q[deps[2]]
//
// Single-pass fusion: walk ll once and accumulate marg1 + marg2 directly
// during the t01 build. For each (s0, s1):
//   sum_t01 = sum_{s2} ll[s0,s1,s2] * q2[s2]   (the t01 element)
//   log_q_d1[s1] += q0[s0] * sum_t01           (folds toward marg1)
//   log_q_d2[s2] += (q0[s0] * q1[s1]) * v      (folds toward marg2, inside s2 loop)
//
// This eliminates the scratch_t12 buffer (size S1*S2) and two post-pass
// axpy_fold_leading_add calls for marg1/marg2. Only marg0 still needs t01
// because its sgemv (sum over s1 of t01[s0,s1]*q1[s1]) can't be folded into
// the inner loop without re-reading per-(s0,s1) values.
//
// Numerics: the accumulation order for log_q_d1 / log_q_d2 changes from a
// single full-buffer fold over S0×S1 (resp. S1×S2) to a streaming partial
// accumulation interleaved with the (s0, s1) loop. Floating-point sums
// reorder; max observed |q - q_ref| under test_fpi_ffi.py's 1e-6 atol stays
// in low-1e-7 territory.
//
// scratch_t01 must be sized >= S0*S1.
// All eight live pointers reference distinct memory: ll_m in ll_flat;
// q0/q1/q2 are non-overlapping slices of scratch_q (distinct lp_offs);
// log_q_d0/log_q_d1/log_q_d2 are non-overlapping slices of scratch_log_q;
// scratch_t01 is its own thread-local buffer. Restrict on q2 and log_q_d2
// is the highest-leverage one (innermost s2 loop reads q2 and writes log_q_d2
// each iter), but the rest pay off in the outer-loop hoisting too.
inline void modality_K3(const float* __restrict__ ll_m, int64_t S0, int64_t S1, int64_t S2,
                        const float* __restrict__ q0, const float* __restrict__ q1, const float* __restrict__ q2,
                        float* __restrict__ log_q_d0, float* __restrict__ log_q_d1, float* __restrict__ log_q_d2,
                        float* __restrict__ scratch_t01) {
  for (int64_t s0 = 0; s0 < S0; ++s0) {
    const float  qs0    = q0[s0];
    const float* ll_s0  = ll_m + s0 * S1 * S2;
    float*       t01_s0 = scratch_t01 + s0 * S1;
    for (int64_t s1 = 0; s1 < S1; ++s1) {
      const float* ll_row  = ll_s0 + s1 * S2;
      const float  q01     = qs0 * q1[s1];
      float        sum_t01 = 0.0f;
      for (int64_t s2 = 0; s2 < S2; ++s2) {
        const float v = ll_row[s2];
        sum_t01 += v * q2[s2];
        log_q_d2[s2] += q01 * v;
      }
      t01_s0[s1] = sum_t01;
      log_q_d1[s1] += qs0 * sum_t01;
    }
  }
  sgemv_rm_f32_add(S0, S1, scratch_t01, S1, q1, log_q_d0);
}

// K>=4 modality (ll has shape Ss[0]×Ss[1]×...×Ss[K-1], row-major):
//
// For each output factor k in [0, K), compute
//   marg_k[s_k] = sum over (others) of ll * prod_{j != k} q_j[s_j]
//              = sum_{s_{k+1}..s_{K-1}} F_k[s_k, s_{k+1}, ..., s_{K-1}]
//                                       * suffix_q[k][s_{k+1}..s_{K-1}]
// where
//   F_k[s_k, ..., s_{K-1}]   = sum_{s_0..s_{k-1}} ll * q_0 * ... * q_{k-1}
//   suffix_q[k][s_{k+1}..]   = q_{k+1} * q_{k+2} * ... * q_{K-1}
//
// Algorithm (this is asymptotically the same cost as JAX/opt_einsum's CSE-ed
// marginals — ~K * prod(S) work per modality per iter, vs the K^2 * prod(S) of
// a naive K-independent leave-one-out):
//   1. Build suffix_q chain (reverse): K-2 outer-product passes.
//   2. Build F_k chain (forward): F_0 = ll; F_{k+1} = axpy_fold_leading over
//      F_k's leading axis weighted by q_k. K-1 passes, sizes shrink by S_k
//      each step → total work ≈ prod(S) * (1 + 1/S_0 + 1/(S_0*S_1) + ...) <
//      2*prod(S).
//   3. Per-k: sgemv_rm_f32_add into log_q_outs[k] using F_k as the matrix and
//      suffix_q[k] as the vector. K passes; per-k cost is prod_{i>=k} S_i.
//
// Memory: F chain + suffix chain back-to-back. Both sized to sum_{k=1..K-1}
// prod_{i>=k} S_i ≤ prod(S). prefix_buf reused as F chain storage; suffix_buf
// holds suffix chain. tmp_buf unused on this path.
//
// Buffers must be sized for the worst-case modality (handled by the caller via
// FpiScratch::ensure_buffers using max_prefix / max_suffix computed once per
// call from A_dependencies).
// ll_m, fchain_buf, suffix_buf are distinct buffers (ll_flat / FpiScratch::prefix /
// FpiScratch::suffix). qs[k] entries are non-overlapping scratch_q slices and
// log_q_outs[k] entries are non-overlapping scratch_log_q slices, both
// guaranteed by the dedup check in FpiCpu — restrict on the array-of-pointers
// declarations is sound. The qs[k] / log_q_outs[k] entries themselves are
// loaded into local pointers inside the loops; we don't need to restrict
// each individually because the called primitives (axpy_fold_leading,
// sgemv_rm_f32_add) already restrict their parameters.
inline void modality_Kn(int K, const float* __restrict__ ll_m, const int32_t* __restrict__ Ss,
                        const int64_t* __restrict__ tail, const int64_t* __restrict__ suf_offs,
                        const int64_t* __restrict__ f_offs, const float* const* __restrict__ qs,
                        float* const* __restrict__ log_q_outs, float* __restrict__ fchain_buf,
                        float* __restrict__ suffix_buf) {
  // Caller-side precondition: K in [4, kMaxFfiDependencyRank]. The dispatch in fpi_one_batch
  // gates this branch on `md.K >= 4` and FpiCpu validates `K <= kMaxFfiDependencyRank`
  // before populating the dispatch table. `__builtin_unreachable` lets the optimizer
  // eliminate the guard entirely AND silences clang-analyzer-security.ArrayBound on
  // the `tail[K - 1]` reads below. Same portable form as build_qs_outer_history_typed.
  if (K < 1 || K > kMaxFfiDependencyRank) __builtin_unreachable();

  // tail[]/suf_offs[]/f_offs[] are precomputed in FpiCpu's per-call dispatch
  // decode (md.tail, md.suf_offs, md.f_offs) — see the K>=4 block there.
  // Hoisting them out of this function takes ~5 muls + ~10 adds × num_iter
  // off the inner loop. Per-k semantics:
  //   tail[k]     = prod_{i>k} Ss[i];  tail[K-1] = 1
  //   suf_offs[k] = offset of suffix_q[k] within suffix_buf, size tail[k]
  //   f_offs[k]   = offset of F_k within fchain_buf, size Ss[k] * tail[k]
  //                 (f_offs[0] / f_offs[1] are 0; F_0 aliases ll directly)

  // ---- 1. Build suffix_q chain (reverse): suffix_q[k] = q_{k+1} ⊗ suffix_q[k+1].
  //   suffix_q[K-1] is implicit scalar 1 (no storage).
  //   suffix_q[K-2] = q_{K-1}.
  if (K >= 2) {
    std::memcpy(suffix_buf + suf_offs[K - 2], qs[K - 1], tail[K - 2] * sizeof(float));
  }
  for (int k = K - 3; k >= 0; --k) {
    const float*  qk1     = qs[k + 1];
    const float*  prev    = suffix_buf + suf_offs[k + 1];
    float*        dst     = suffix_buf + suf_offs[k];
    const int64_t qsz     = Ss[k + 1];
    const int64_t prev_sz = tail[k + 1];
    // dst[i, j] = qk1[i] * prev[j], row-major over (i, j).
    for (int64_t i = 0; i < qsz; ++i) {
      const float qi  = qk1[i];
      float*      row = dst + i * prev_sz;
      for (int64_t j = 0; j < prev_sz; ++j) row[j] = qi * prev[j];
    }
  }

  // ---- 2. Build F_k chain (forward): F_{k+1}[s_{k+1}..] = sum_{s_k} F_k[s_k..] * q_k[s_k].
  //   F_0 aliases ll. F_k has shape (Ss[k], tail[k]) viewed as (lead, inner)
  //   so axpy_fold_leading collapses the leading axis weighted by q_k.
  for (int k = 0; k < K - 1; ++k) {
    const float*  fk_in  = (k == 0) ? ll_m : (fchain_buf + f_offs[k]);
    float*        fk_out = fchain_buf + f_offs[k + 1];
    const int64_t lead   = Ss[k];
    const int64_t inner  = tail[k];  // = Ss[k+1] * tail[k+1] = size of F_{k+1}.
    axpy_fold_leading(lead, inner, fk_in, qs[k], fk_out);
  }

  // ---- 3. Per-k marginal accumulation:
  //   marg_k[s_k] = sum over trailing dims of F_k[s_k, trailing] * suffix_q[k][trailing].
  //   Use sgemv_rm_f32_add to write straight into log_q_outs[k].
  for (int k = 0; k < K; ++k) {
    const float*  fk    = (k == 0) ? ll_m : (fchain_buf + f_offs[k]);
    const int64_t Sk    = Ss[k];
    const int64_t inner = tail[k];
    if (inner == 1) {
      // suffix_q[K-1] is scalar 1; marg_{K-1}[s_k] = F_{K-1}[s_k]. Add directly.
      float* out = log_q_outs[k];
      for (int64_t s = 0; s < Sk; ++s) out[s] += fk[s];
    } else {
      // sgemv_rm_f32_add: m=Sk rows × n=inner cols, A=F_k (lda=inner),
      //                   x=suffix_q[k], y=log_q_outs[k] (accumulating).
      sgemv_rm_f32_add(Sk, inner, fk, inner, suffix_buf + suf_offs[k], log_q_outs[k]);
    }
  }
}

// =============================================================================
// Per-call types & scratch
// =============================================================================

// Per-modality dispatch info resolved once in FpiCpu so the num_iter * M hot
// loop can jump straight to modality_K{1,2,3} or modality_Kn (K>=4).
//
// For K<=3, S0/S1/S2 + lp_off0/1/2 hold the per-position state size and log_q
// offset (the K=1/K=2/K=3 paths read these directly).
//
// For K>=4, Ss[i] / lp_offs[i] (i in [0, K)) carry the same info for the
// generic modality_Kn path; S0/S1/S2 / lp_off0/1/2 are mirrored from Ss[0..2]
// for the K=1/K=2/K=3 hot paths.
struct ModalityDispatch {
  int K;  // 1..kMaxFfiDependencyRank
  // Ss[i] is a per-factor state size, lp_offs[i] is an offset into the per-call
  // log_q buffer. Both fit int32_t for any practical pymdp model (factor states
  // ~hundreds, total_S sum-of-factor-states ~thousands), and narrowing halves
  // the per-modality footprint of the K=1/K=2/K=3 hot path so it lands in one
  // 64-byte cache line on Cortex-A57.
  std::array<int32_t, kMaxFfiDependencyRank> Ss;       // S[A_deps[m][i]] for i in [0, K)
  std::array<int32_t, kMaxFfiDependencyRank> lp_offs;  // lp_offsets[A_deps[m][i]] for i in [0, K)
  int64_t                                    ll_off;   // ll_offsets[m]
  // K>=4 only: precomputed by FpiCpu, read by modality_Kn each iter:
  //   tail[k]     = prod_{i>k} Ss[i],  tail[K-1] = 1
  //   suf_offs[k] = offset into suffix_buf for suffix_q[k]
  //   f_offs[k]   = offset into fchain_buf for F_k (f_offs[0]/f_offs[1] = 0)
  // Hoisted out of modality_Kn so the num_iter * M call site doesn't redo
  // the K-pass computation each invocation. Ignored for K<=3 hot paths. Stay
  // int64_t — products of state sizes can overflow int32_t at the rank-8 ABI cap.
  std::array<int64_t, kMaxFfiDependencyRank> tail;
  std::array<int64_t, kMaxFfiDependencyRank> suf_offs;
  std::array<int64_t, kMaxFfiDependencyRank> f_offs;
};

// Per-call scratch with two distinct usage patterns:
//
//   1. mods (dispatch table): built once on the master thread BEFORE the
//      parallel region. Workers access the master's mods via the const
//      pointer passed into fpi_one_batch — TLS is regular memory and the
//      OMP barrier ensures the master outlives all readers.
//   2. log_q / q / t01 / t12 (compute scratch): each OMP worker calls
//      ensure_buffers on its OWN g_fpi_scratch inside the parallel region
//      and uses its own buffers. No false sharing or cross-worker writes.
//
// Resize-up-only; pymdp's call patterns repeat the same shapes for many
// calls, so the steady-state allocation count is zero after warm-up — that
// holds per-thread too, since libgomp pools workers across calls.
//
// Batch parallelism uses a two-tier gate (see fire_batch_parallel below):
//   - `batch >= kOmpNodeThreshold` for the high-batch path (jax-perf bench
//     fixtures with large vmap), and
//   - a work-aware lower gate (`batch >= 2 && ni * total_ll >= 100000`) that
//     fires for production rollout (batch=4, fpi_large-shape) where per-batch
//     FPI work amortizes the libgomp/libomp fork+barrier.

// Bundle of distinct compute-scratch buffer pointers carved out of FpiScratch.
// `__restrict__` on each member guarantees no-alias across all members — every
// member maps to its own std::vector allocation in FpiScratch.
struct FpiScratchPtrs {
  float* __restrict__ log_q;
  float* __restrict__ log_q_prev;
  float* __restrict__ q;
  float* __restrict__ t01;
  float* __restrict__ prefix;
  float* __restrict__ suffix;
};

struct FpiScratch {
  std::vector<float>            log_q;
  std::vector<float>            log_q_prev;  // convergence-check snapshot of prior iter's log_q
  std::vector<float>            q;
  std::vector<float>            t01;     // K=3 shared prefix (still used for marg0 sgemv)
  std::vector<float>            prefix;  // K>=4 F-chain storage (extended in-place)
  std::vector<float>            suffix;  // K>=4 suffix_q tensors back-to-back
  std::vector<ModalityDispatch> mods;

  void ensure_dispatch(int64_t M) { ensure_at_least(mods, M); }
  void ensure_buffers(int64_t total_S, int64_t max_t01, int64_t max_prefix, int64_t max_suffix) {
    ensure_at_least(log_q, total_S);
    ensure_at_least(log_q_prev, total_S);
    ensure_at_least(q, total_S);
    ensure_at_least(t01, max_t01);
    ensure_at_least(prefix, max_prefix);
    ensure_at_least(suffix, max_suffix);
  }

  FpiScratchPtrs as_ptrs() {
    return {log_q.data(), log_q_prev.data(), q.data(), t01.data(), prefix.data(), suffix.data()};
  }
};
inline thread_local FpiScratch g_fpi_scratch{};

// =============================================================================
// Single-batch driver
// =============================================================================

// max |a[i] - b[i]| over [0, n). Used by the convergence-check path. Scalar
// is fine — total_S is small (~72 floats for production) and the kernel does
// at most num_iter passes. Auto-vectorizes to AVX2/NEON max-of-abs-diff.
inline float max_abs_diff(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// Hard-coded early-stop tolerance: after each body iter, kernel computes
// `max|log_q - log_q_prev|` and breaks once below this. Empirically chosen
// at 1e-5 — well below test_fpi_ffi.py's parity atol of 1e-6 (max observed
// |q - q_ref| was 2.7e-7 across all fixtures), still loose enough to fire
// on small/easy shapes (fpi_inference, fpi_high_rank). Larger problems
// (fpi_large) don't converge to this threshold within typical num_iter=16
// and run the full loop. Cost when not firing: one memcpy + max-abs-diff
// per iter, ~5% of per-iter cost — recouped many times over when it does.
constexpr float kFpiConvergenceTol = 1e-5f;

// Validate attr-side invariants: F/M/num_iter positive, span sizes match
// F+1 / M+1, S[f] positive, and lp/ll/A_dep offsets monotonic. Pure attr work
// — no buffer reads.
inline FfiError validate_fpi_attrs(FfiInt64Span S, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets,
                                   FfiInt64Span A_dep_offsets, int32_t num_iter, int64_t F, int64_t M) {
  if (F <= 0 || M <= 0) {
    return invalid_arg(kFpiKernelName, "invalid F=" + std::to_string(F) + " or M=" + std::to_string(M));
  }
  PYMDP_TRY(check_span_size(kFpiKernelName, "lp_offsets", static_cast<int64_t>(lp_offsets.size()), F + 1));
  PYMDP_TRY(check_span_size(kFpiKernelName, "ll_offsets", static_cast<int64_t>(ll_offsets.size()), M + 1));
  if (num_iter <= 0) {
    return invalid_arg(kFpiKernelName, "num_iter = " + std::to_string(num_iter) + ", must be positive");
  }
  for (int64_t f = 0; f < F; ++f) {
    if (S[f] <= 0) {
      return invalid_arg(kFpiKernelName, "S[" + std::to_string(f) + "] must be positive");
    }
    PYMDP_TRY(check_monotonic(kFpiKernelName, "lp_offsets", lp_offsets[f], lp_offsets[f + 1]));
  }
  for (int64_t m = 0; m < M; ++m) {
    PYMDP_TRY(check_monotonic(kFpiKernelName, "ll_offsets", ll_offsets[m], ll_offsets[m + 1]));
    PYMDP_TRY(check_monotonic(kFpiKernelName, "A_dep_offsets", A_dep_offsets[m], A_dep_offsets[m + 1]));
  }
  return FfiError::Success();
}

// Build the per-modality dispatch table. Validates each modality's dep rank
// and factor refs, populates the K hot-path fields (Ss, lp_offs, ll_off), and
// for K>=4 precomputes the (tail, suf_offs, f_offs) triples used by
// modality_Kn so the num_iter*M hot loop doesn't redo the scalar setup.
// Reports the worst-case scratch sizes (max_t01 / max_prefix / max_suffix) via
// out-params for FpiScratch::ensure_buffers.
inline FfiError build_modality_dispatch(FfiInt64Span S, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets,
                                        FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int64_t F, int64_t M,
                                        std::vector<ModalityDispatch>* mods, int64_t* max_t01, int64_t* max_prefix,
                                        int64_t* max_suffix) {
  *max_t01    = 0;
  *max_prefix = 0;
  *max_suffix = 0;
  for (int64_t m = 0; m < M; ++m) {
    const int64_t     dep_start = A_dep_offsets[m];
    const int64_t     K         = A_dep_offsets[m + 1] - dep_start;
    ModalityDispatch& md        = (*mods)[m];
    md.K                        = static_cast<int>(K);
    md.ll_off                   = ll_offsets[m];
    md.Ss.fill(0);
    md.lp_offs.fill(-1);
    if (K < 1 || K > kMaxFfiDependencyRank) {
      return invalid_arg(kFpiKernelName, "modality " + std::to_string(m) + " has dep rank " + std::to_string(K) +
                                             " (must be in [1, " + std::to_string(kMaxFfiDependencyRank) + "])");
    }
    for (int64_t i = 0; i < K; ++i) {
      const int64_t d = A_dep_flat[dep_start + i];
      if (d < 0 || d >= F) {
        return invalid_arg(kFpiKernelName, "modality " + std::to_string(m) + " references out-of-range factor");
      }
      // Distinct factors per modality: the hot-loop kernels mark q[deps[i]] /
      // log_q[deps[i]] slices __restrict__, so aliasing would be silent UB.
      // Python's can_handle_fpi rejects duplicates up front; this is the C++
      // re-check at the ABI boundary.
      for (int64_t j = 0; j < i; ++j) {
        if (md.lp_offs[j] == lp_offsets[d]) {
          return invalid_arg(kFpiKernelName,
                             "modality " + std::to_string(m) + " has duplicate factor in A_dependencies");
        }
      }
      md.Ss[i]      = static_cast<int32_t>(S[d]);
      md.lp_offs[i] = static_cast<int32_t>(lp_offsets[d]);
    }
    if (K == 3) {
      *max_t01 = std::max<int64_t>(*max_t01, static_cast<int64_t>(md.Ss[0]) * md.Ss[1]);
    }
    if (K >= 4) {
      // tail[k]     = prod_{i>k} Ss[i],  tail[K-1] = 1
      // suf_offs[k] = offset into suffix_buf for suffix_q[k] (size tail[k])
      // f_offs[k]   = offset into fchain_buf for F_k (size Ss[k]*tail[k]);
      //               f_offs[0]/f_offs[1] are 0 (F_0 aliases ll directly).
      md.tail[K - 1]     = 1;
      md.suf_offs[K - 1] = 0;
      for (int k = K - 2; k >= 0; --k) {
        md.tail[k]     = md.tail[k + 1] * md.Ss[k + 1];
        md.suf_offs[k] = md.suf_offs[k + 1] + md.tail[k + 1];
      }
      md.f_offs[0] = 0;
      if (K >= 2) md.f_offs[1] = 0;
      for (int k = 2; k < K; ++k) {
        md.f_offs[k] = md.f_offs[k - 1] + md.Ss[k - 1] * md.tail[k - 1];
      }
      const int64_t fchain_total = md.f_offs[K - 1] + md.Ss[K - 1] * md.tail[K - 1];
      const int64_t suffix_total = md.suf_offs[0] + md.tail[0];
      *max_prefix                = std::max<int64_t>(*max_prefix, fchain_total);
      *max_suffix                = std::max<int64_t>(*max_suffix, suffix_total);
    }
  }
  return FfiError::Success();
}

// Single-batch FPI: runs num_iter iterations and writes the final softmax to
// q_out (flat, layout matches lp_offsets).
// All buffer pointers reference distinct memory:
//   * lp_flat / ll_flat are caller-owned XLA inputs (different buffers)
//   * q_out is the caller-owned XLA output (different from inputs)
//   * S / lp_offsets are caller-owned attribute spans (separate from buffers)
//   * mods is a thread_local-owned dispatch table read-only inside this call
//   * scratch_* are thread_local FpiScratch fields (each its own std::vector
//     allocation) — non-overlapping with each other and with the XLA buffers
inline void fpi_one_batch(int64_t num_iter, int64_t F, int64_t M, int64_t total_S, const int64_t* __restrict__ S,
                          const int64_t* __restrict__ lp_offsets, const float* __restrict__ lp_flat,
                          const float* __restrict__ ll_flat, const ModalityDispatch* __restrict__ mods,
                          float* __restrict__ q_out, const FpiScratchPtrs& sc) {
  // Caller-validated invariants — restate so the optimizer can drop zero-trip
  // guards on per-F / per-M / per-total_S loops in this body.
  if (F <= 0 || M <= 0 || total_S <= 0 || num_iter <= 0) __builtin_unreachable();

  // Re-alias struct members to local names for readability. `__restrict__ const`
  // preserves the no-alias contract: each maps to a distinct FpiScratch vector.
  float* __restrict__ const scratch_log_q      = sc.log_q;
  float* __restrict__ const scratch_log_q_prev = sc.log_q_prev;
  float* __restrict__ const scratch_q          = sc.q;
  float* __restrict__ const scratch_t01        = sc.t01;
  float* __restrict__ const scratch_prefix     = sc.prefix;
  float* __restrict__ const scratch_suffix     = sc.suffix;

  // log_q starts at zero, matching the JAX scan reference in run_factorized_fpi.
  std::memset(scratch_log_q, 0, total_S * sizeof(float));

  for (int64_t it = 0; it < num_iter; ++it) {
    // q[f] = softmax(log_q[f]) AND snapshot log_q -> log_q_prev for the
    // convergence check below. The snapshot is folded into the softmax
    // pass — same single load of scratch_log_q feeds both writes — so we
    // pay one full-buffer pass per iter instead of two.
    softmax_per_factor_snapshot(F, S, lp_offsets, scratch_log_q, scratch_q, scratch_log_q_prev);

    // Reset accumulator: log_q[f] = log_prior[f]
    std::memcpy(scratch_log_q, lp_flat, total_S * sizeof(float));

    // Accumulate per-modality marginals into log_q.
    for (int64_t m = 0; m < M; ++m) {
      const ModalityDispatch& md   = mods[m];
      const float*            ll_m = ll_flat + md.ll_off;

      if (md.K == 1) {
        modality_K1(ll_m, md.Ss[0], scratch_log_q + md.lp_offs[0]);
      } else if (md.K == 2) {
        modality_K2(ll_m, md.Ss[0], md.Ss[1], scratch_q + md.lp_offs[0], scratch_q + md.lp_offs[1],
                    scratch_log_q + md.lp_offs[0], scratch_log_q + md.lp_offs[1]);
      } else if (md.K == 3) {
        modality_K3(ll_m, md.Ss[0], md.Ss[1], md.Ss[2], scratch_q + md.lp_offs[0], scratch_q + md.lp_offs[1],
                    scratch_q + md.lp_offs[2], scratch_log_q + md.lp_offs[0], scratch_log_q + md.lp_offs[1],
                    scratch_log_q + md.lp_offs[2], scratch_t01);
      } else {  // md.K in [4, kMaxFfiDependencyRank]
        std::array<const float*, kMaxFfiDependencyRank> qs_ptrs{};
        std::array<float*, kMaxFfiDependencyRank>       log_q_outs{};
        for (int i = 0; i < md.K; ++i) {
          qs_ptrs[i]    = scratch_q + md.lp_offs[i];
          log_q_outs[i] = scratch_log_q + md.lp_offs[i];
        }
        modality_Kn(md.K, ll_m, md.Ss.data(), md.tail.data(), md.suf_offs.data(), md.f_offs.data(), qs_ptrs.data(),
                    log_q_outs.data(), scratch_prefix, scratch_suffix);
      }
    }

    // Skip the convergence check on iter 0: log_q_prev was a memset(0) and
    // log_q is now lp + sum_m marg_m(uniform), so the delta is large by
    // construction and the early-exit can't fire.
    if (it > 0) {
      const float delta = max_abs_diff(scratch_log_q, scratch_log_q_prev, total_S);
      if (delta < kFpiConvergenceTol) break;
    }
  }

  softmax_per_factor(F, S, lp_offsets, scratch_log_q, q_out);
}

}  // namespace

// Shared kernel body. Both FpiCpu (platform="cpu", host buffers) and
// FpiCudaHost (platform="CUDA", device buffers D2H'd into host scratch)
// funnel through here so the validated K=1/2/3 hot paths + K>=4 forward-chain
// generic path + batch OMP path are written exactly once. Inputs/outputs are
// raw host pointers + counts.
FfiError run_fpi_kernel_host(const float* ll_flat, int64_t ll_count, const float* lp_flat, int64_t lp_count,
                             float* q_out, int64_t q_count, FfiInt64Span S, FfiInt64Span ll_offsets,
                             FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets,
                             int32_t num_iter) {
  const int64_t F = static_cast<int64_t>(S.size());
  const int64_t M = static_cast<int64_t>(A_dep_offsets.size()) - 1;
  PYMDP_TRY(validate_fpi_attrs(S, ll_offsets, lp_offsets, A_dep_offsets, num_iter, F, M));

  const int64_t total_S  = lp_offsets[F];
  const int64_t total_ll = ll_offsets[M];
  if (total_S <= 0 || lp_count <= 0 || lp_count % total_S != 0) {
    return invalid_arg(kFpiKernelName, "lp_flat size = " + std::to_string(lp_count) +
                                           " not divisible by total_S = " + std::to_string(total_S));
  }
  const int64_t batch = lp_count / total_S;
  PYMDP_TRY(check_count(kFpiKernelName, "ll_flat", ll_count, batch * total_ll));
  PYMDP_TRY(check_count(kFpiKernelName, "q_out", q_count, batch * total_S));

  // Caller-validated invariants — restate so the optimizer can drop zero-trip
  // guards on the per-batch / per-factor / per-modality loops below.
  if (F <= 0 || M <= 0 || batch <= 0 || total_S <= 0 || num_iter <= 0) __builtin_unreachable();

  // mods table + scratch buffers backed by thread_local g_fpi_scratch so
  // repeated FPI calls of the same shape allocate zero times after warm-up.
  FpiScratch& sc = g_fpi_scratch;
  sc.ensure_dispatch(M);
  int64_t max_t01 = 0, max_prefix = 0, max_suffix = 0;
  PYMDP_TRY(build_modality_dispatch(S, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets, F, M, &sc.mods, &max_t01,
                                    &max_prefix, &max_suffix));

  KernelTimer timer([&](double us) {
    std::fprintf(stderr, "[fpi] batch=%lld F=%lld M=%lld iter=%d total_S=%lld | total=%6.2f us\n",
                 static_cast<long long>(batch), static_cast<long long>(F), static_cast<long long>(M),
                 static_cast<int>(num_iter), static_cast<long long>(total_S), us);
  });

  // Dispatch one batch element. Body is identical between the serial and
  // parallel paths; the parallel split is structural (skips libomp fork for
  // the small-batch path entirely — an `omp parallel if(false)` still spins
  // up a 1-thread team).
  const ModalityDispatch* mods_ptr = sc.mods.data();
  const int64_t           ni       = num_iter;
  auto                    run_one  = [&](int64_t b, const FpiScratchPtrs& sc_ptrs) {
    fpi_one_batch(ni, F, M, total_S, S.begin(), lp_offsets.begin(), lp_flat + b * total_S, ll_flat + b * total_ll,
                  mods_ptr, q_out + b * total_S, sc_ptrs);
  };

  if (!should_parallelize_fpi_batch(batch, ni, total_ll)) {
    g_fpi_scratch.ensure_buffers(total_S, max_t01, max_prefix, max_suffix);
    const FpiScratchPtrs sc_ptrs = g_fpi_scratch.as_ptrs();
    for (int64_t b = 0; b < batch; ++b) run_one(b, sc_ptrs);
  } else {
#pragma omp parallel num_threads(omp_team_size_for_work_units(batch))
    {
      g_fpi_scratch.ensure_buffers(total_S, max_t01, max_prefix, max_suffix);
      const FpiScratchPtrs sc_ptrs = g_fpi_scratch.as_ptrs();
#pragma omp for schedule(static)
      for (int64_t b = 0; b < batch; ++b) run_one(b, sc_ptrs);
    }
  }

  return FfiError::Success();
}

// =============================================================================
// ABI entry points
// =============================================================================

FfiError FpiCpu(FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S, FfiInt64Span ll_offsets,
                FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter) {
  return run_fpi_kernel_host(ll_flat.typed_data(), ll_flat.element_count(), lp_flat.typed_data(),
                             lp_flat.element_count(), q_out->typed_data(), q_out->element_count(), S, ll_offsets,
                             lp_offsets, A_dep_flat, A_dep_offsets, num_iter);
}

#ifdef PYMDP_FFI_HAS_CUDA

// platform="CUDA" shim. JAX hands us device-pointer Buffers + the stream;
// we D2H ll/lp into host scratch, run the validated CPU kernel, then H2D
// q_out back to JAX's device buffer. Used when JAX's default backend is
// CUDA — FPI has no native CUDA implementation, but the CPU kernel beats
// the JAX-scan-on-GPU fallback (per-iter launch overhead × num_iter dwarfs
// the D2H/H2D pair for FPI's tiny shapes; see fpi_inference profile).
//
// Thread-local scratch keeps per-call allocations to a resize (capacity
// retained across calls of the same shape).
FfiError FpiCudaHost(cudaStream_t stream, FfiF32Buf ll_flat_dev, FfiF32Buf lp_flat_dev, FfiF32Out q_out_dev,
                     FfiInt64Span S, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                     FfiInt64Span A_dep_offsets, int32_t num_iter) {
  const int64_t ll_count = ll_flat_dev.element_count();
  const int64_t lp_count = lp_flat_dev.element_count();
  const int64_t q_count  = q_out_dev->element_count();

  // Tegra zero-copy fast path. On integrated GPUs JAX typically allocates
  // device buffers as managed or pinned memory, both host-accessible. When
  // every non-empty buffer aliases, we skip the cudaMemcpyAsync trio entirely
  // — read ll/lp directly from JAX's buffer, write q_out into JAX's buffer,
  // and return. The stream sync below is still required (we must wait for
  // prior ops on the stream to finish writing ll/lp before reading them) but
  // the post-H2D round-trip and the per-launch overhead of three memcpy ops
  // disappear. Subsequent ops JAX queues on this stream see q_out written
  // because cudaLaunchKernel establishes a happens-before from any prior host
  // writes (managed/pinned memory is hardware-coherent on Tegra).
  const float* ll_alias =
      ll_count > 0 ? static_cast<const float*>(try_alias_as_host(ll_flat_dev.typed_data())) : nullptr;
  const float* lp_alias =
      lp_count > 0 ? static_cast<const float*>(try_alias_as_host(lp_flat_dev.typed_data())) : nullptr;
  float*     q_alias   = q_count > 0 ? static_cast<float*>(try_alias_as_host(q_out_dev->typed_data())) : nullptr;
  const bool can_alias = (ll_count == 0 || ll_alias != nullptr) && (lp_count == 0 || lp_alias != nullptr) &&
                         (q_count == 0 || q_alias != nullptr);

  if (can_alias) {
    if (cudaError_t rc = cudaStreamSynchronize(stream); rc != cudaSuccess) {
      return invalid_arg(kFpiKernelName,
                         std::string("cudaStreamSynchronize (alias path) failed: ") + cudaGetErrorString(rc));
    }
    return run_fpi_kernel_host(ll_alias, ll_count, lp_alias, lp_count, q_alias, q_count, S, ll_offsets, lp_offsets,
                               A_dep_flat, A_dep_offsets, num_iter);
  }

  // Discrete-GPU fallback: D2H inputs into thread_local scratch, run the
  // kernel, H2D the output. Used when JAX's allocator returns device-only
  // memory (cudaMemoryTypeDevice) — typical on dGPU hosts.
  thread_local std::vector<float> ll_host;
  thread_local std::vector<float> lp_host;
  thread_local std::vector<float> q_host;
  ll_host.resize(static_cast<size_t>(std::max<int64_t>(ll_count, 0)));
  lp_host.resize(static_cast<size_t>(std::max<int64_t>(lp_count, 0)));
  q_host.resize(static_cast<size_t>(std::max<int64_t>(q_count, 0)));

  if (ll_count > 0) {
    if (cudaError_t rc = cudaMemcpyAsync(ll_host.data(), ll_flat_dev.typed_data(),
                                         static_cast<size_t>(ll_count) * sizeof(float), cudaMemcpyDeviceToHost, stream);
        rc != cudaSuccess) {
      return invalid_arg(kFpiKernelName, std::string("D2H ll_flat failed: ") + cudaGetErrorString(rc));
    }
  }
  if (lp_count > 0) {
    if (cudaError_t rc = cudaMemcpyAsync(lp_host.data(), lp_flat_dev.typed_data(),
                                         static_cast<size_t>(lp_count) * sizeof(float), cudaMemcpyDeviceToHost, stream);
        rc != cudaSuccess) {
      return invalid_arg(kFpiKernelName, std::string("D2H lp_flat failed: ") + cudaGetErrorString(rc));
    }
  }
  if (cudaError_t rc = cudaStreamSynchronize(stream); rc != cudaSuccess) {
    return invalid_arg(kFpiKernelName,
                       std::string("cudaStreamSynchronize after D2H failed: ") + cudaGetErrorString(rc));
  }

  PYMDP_TRY(run_fpi_kernel_host(ll_host.data(), ll_count, lp_host.data(), lp_count, q_host.data(), q_count, S,
                                ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets, num_iter));

  if (q_count > 0) {
    if (cudaError_t rc = cudaMemcpyAsync(q_out_dev->typed_data(), q_host.data(),
                                         static_cast<size_t>(q_count) * sizeof(float), cudaMemcpyHostToDevice, stream);
        rc != cudaSuccess) {
      return invalid_arg(kFpiKernelName, std::string("H2D q_out failed: ") + cudaGetErrorString(rc));
    }
  }
  // No post-H2D sync needed: stream ordering guarantees any op JAX queues
  // after our return on this stream sees q_out written. Host-side correctness
  // is also covered — pageable H2D is implicitly synchronous wrt the source
  // host buffer (cudaMemcpyAsync stages from `q_host` before returning), so
  // the thread_local resize on the next call doesn't race the in-flight DMA.
  return FfiError::Success();
}

// =============================================================================
// FpiCudaDevice — native CUDA FPI (platform="CUDA")
// =============================================================================
//
// Single CUDA kernel runs all `num_iter` iterations internally, one block per
// batch element. No D2H, no H2D, no host-side stream sync — JAX's surrounding
// ops on the same stream pipeline naturally with the FPI kernel.
//
// Restrictions: every modality's A_dependencies rank must be in [1, 3]. The
// host-side gate in pymdp/ffi/_fpi.py enforces this; calls landing here with
// K>=4 hit the kernel's switch default and produce silently wrong output (the
// gate is the contract). K>=4 dispatches go through FpiCudaHost (shim) or the
// JAX scan reference instead.
//
// Per-call host work: build the dispatch table on host, copy three small
// arrays to device-side scratch (S, lp_offsets, mods — total ~F*4 + F*4 +
// M*16 bytes for K<=3), launch the kernel. All async on the stream so JAX's
// upstream ops can still queue ahead.

namespace {

// Per-thread scratch holding the device-side dispatch arrays. Sized lazily
// via CuArr::ensure (resize-up-only); after warm-up steady-state cost is
// zero allocation.
//
// `sig` is a content fingerprint of the *raw attr spans* used to populate
// the device buffers — S_span, ll_offsets, lp_offsets, A_dep_flat,
// A_dep_offsets, plus an F/M shape tag. Hashing the raw attrs (instead of
// the built host-side dispatch table) means cache hits skip both the H2D
// uploads *and* validate_fpi_attrs + the per-modality dispatch build on
// the host. Within a single rollout the model metadata is invariant, so
// calls 2..N hit and run only the runtime-shape checks (batch derivation)
// before the kernel launch.
//
// `0` is reserved for "never uploaded yet" so a fresh-process first call
// always misses the cache. We use a 64-bit FNV-1a hash and treat the
// astronomical-but-finite collision risk as acceptable: a hash collision
// would silently reuse stale device buffers, but the input space is attr
// spans from real pymdp models, not adversarial data.
struct FpiCudaDeviceScratch {
  // Pointer-fed path (F > kMaxFSmallMeta or M > kMaxMSmallMeta): device
  // buffers populated via H2D on cache miss, kernel reads through global
  // pointers (LDG, L1-cached).
  CuArr S_dev;           // int32_t[F]
  CuArr lp_offsets_dev;  // int32_t[F]
  CuArr mods_dev;        // fpi_cuda::ModalityDispatchGpu[M]
  // Smallmeta path (F <= kMaxFSmallMeta && M <= kMaxMSmallMeta): cached
  // by-value struct passed as a kernel argument, served from the cmem
  // parameter bank (LDC, broadcast). Populated on the same cache-miss
  // event as the device buffers; the active path at launch time is
  // chosen by the F/M dimensions, not by which storage was filled.
  fpi_cuda::FpiSmallMeta smallmeta{};
  uint64_t               sig = 0;

  // Pointer-identity fast cache. XLA stores the static FFI attr arrays in a
  // per-executable buffer that is pointer-stable across calls of the same
  // compiled function — so within a rollout the attr spans handed to us
  // alias the same backing memory each call. Comparing pointers + sizes
  // (~6 word loads + 6 compares) skips the FNV-1a pass entirely on a hit.
  // On miss we fall through to the FNV path, which still validates the
  // content cache via `sig`. Strictly additive: any divergence in attr
  // pointers or sizes routes to the existing miss path with the same
  // semantics it had before.
  const int64_t* last_S_ptr              = nullptr;
  const int64_t* last_ll_offsets_ptr     = nullptr;
  const int64_t* last_lp_offsets_ptr     = nullptr;
  const int64_t* last_A_dep_flat_ptr     = nullptr;
  const int64_t* last_A_dep_offsets_ptr  = nullptr;
  size_t         last_S_size             = 0;
  size_t         last_ll_offsets_size    = 0;
  size_t         last_lp_offsets_size    = 0;
  size_t         last_A_dep_flat_size    = 0;
  size_t         last_A_dep_offsets_size = 0;
};

inline uint64_t fnv1a64(const void* data, size_t bytes, uint64_t seed = 0xcbf29ce484222325ULL) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t       h = seed;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= 0x100000001b3ULL;
  }
  return h;
}
inline thread_local FpiCudaDeviceScratch g_fpi_cuda_dev_scratch;

}  // namespace

FfiError FpiCudaDevice(cudaStream_t stream, FfiF32Buf ll_flat_dev, FfiF32Buf lp_flat_dev, FfiF32Out q_out_dev,
                       FfiInt64Span S_span, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                       FfiInt64Span A_dep_offsets, int32_t num_iter) {
  const int64_t F = static_cast<int64_t>(S_span.size());
  const int64_t M = static_cast<int64_t>(A_dep_offsets.size()) - 1;

  // Cheap span-bounds guard so the sig-hash below can read the spans
  // without UB. The full attr validation (monotonicity, positivity, K
  // range) is gated by the cache miss path; cache hits skip it.
  if (F <= 0 || M <= 0) {
    return invalid_arg(kFpiKernelName, "invalid F=" + std::to_string(F) + " or M=" + std::to_string(M));
  }
  if (static_cast<int64_t>(lp_offsets.size()) != F + 1 || static_cast<int64_t>(ll_offsets.size()) != M + 1) {
    return invalid_arg(kFpiKernelName, "lp_offsets/ll_offsets span size mismatch with F/M");
  }

  // Two-tier host-side metadata cache:
  //   1. Pointer-identity fast path: XLA's static-attr buffer is pointer-
  //      stable across calls of the same compiled executable. Compare the
  //      five attr-span begin pointers + sizes against the last upload. On
  //      hit we skip the FNV-1a pass entirely.
  //   2. FNV-1a content hash (fallback): runs only when pointers/sizes
  //      diverge. Hashes the raw attr spans (S, ll_offsets, lp_offsets,
  //      A_dep_flat, A_dep_offsets, plus F/M shape tag) and short-circuits
  //      when it matches the last upload. Hashing raw attrs (instead of
  //      the built host dispatch table) lets us skip the build entirely
  //      on a hit — plus validate_fpi_attrs and the per-modality
  //      K/factor-range checks, since those would all pass again with
  //      byte-identical inputs.
  //
  // Within a single rollout the model metadata is invariant *and* the attr
  // pointers are stable, so calls 2..N take the pointer-identity hit and
  // run the bare minimum host work before the launch.
  const FpiCudaDeviceScratch& cs = g_fpi_cuda_dev_scratch;
  const bool                  ptr_cache_hit =
      cs.sig != 0 && cs.last_S_ptr == S_span.begin() && cs.last_S_size == S_span.size() &&
      cs.last_ll_offsets_ptr == ll_offsets.begin() && cs.last_ll_offsets_size == ll_offsets.size() &&
      cs.last_lp_offsets_ptr == lp_offsets.begin() && cs.last_lp_offsets_size == lp_offsets.size() &&
      cs.last_A_dep_flat_ptr == A_dep_flat.begin() && cs.last_A_dep_flat_size == A_dep_flat.size() &&
      cs.last_A_dep_offsets_ptr == A_dep_offsets.begin() && cs.last_A_dep_offsets_size == A_dep_offsets.size();

  uint64_t sig;
  bool     host_cache_hit;
  if (ptr_cache_hit) {
    sig            = cs.sig;
    host_cache_hit = true;
  } else {
    sig = fnv1a64(S_span.begin(), static_cast<size_t>(S_span.size()) * sizeof(int64_t));
    sig = fnv1a64(ll_offsets.begin(), static_cast<size_t>(ll_offsets.size()) * sizeof(int64_t), sig);
    sig = fnv1a64(lp_offsets.begin(), static_cast<size_t>(lp_offsets.size()) * sizeof(int64_t), sig);
    sig = fnv1a64(A_dep_flat.begin(), static_cast<size_t>(A_dep_flat.size()) * sizeof(int64_t), sig);
    sig = fnv1a64(A_dep_offsets.begin(), static_cast<size_t>(A_dep_offsets.size()) * sizeof(int64_t), sig);
    // Mix F and M in too — defends against pathological cases where two
    // distinct (F, M) shapes hash equivalently on their prefixes.
    const uint64_t shape_tag = (static_cast<uint64_t>(F) << 32) ^ static_cast<uint64_t>(M);
    sig                      = fnv1a64(&shape_tag, sizeof(shape_tag), sig);

    // `sig == 0` is reserved for "never uploaded yet" so the initial
    // FpiCudaDeviceScratch state never spuriously hits. fnv1a64's output
    // is virtually never zero for non-empty input (probability ~2^-64),
    // and even a hypothetical collision would just be treated as a miss
    // — re-validate, re-build, re-upload, set sig. Correctness-safe.
    host_cache_hit = (sig != 0) && (sig == cs.sig);
  }

  if (!host_cache_hit) {
    PYMDP_TRY(validate_fpi_attrs(S_span, ll_offsets, lp_offsets, A_dep_offsets, num_iter, F, M));
  } else if (num_iter <= 0) {
    // num_iter isn't part of the cache key (it doesn't affect the dispatch
    // table), so guard it explicitly here for the hit path. validate_fpi_attrs
    // covers it on the miss path.
    return invalid_arg(kFpiKernelName, "num_iter = " + std::to_string(num_iter) + ", must be positive");
  }

  // Runtime-shape checks: always run. lp_flat_dev / ll_flat_dev / q_out_dev
  // element counts come from the JAX runtime (batch dim under vmap) and are
  // not part of the static attrs — same model metadata may be called with
  // different batch sizes.
  const int64_t total_S  = lp_offsets[F];
  const int64_t total_ll = ll_offsets[M];
  const int64_t lp_count = lp_flat_dev.element_count();
  if (total_S <= 0 || lp_count <= 0 || lp_count % total_S != 0) {
    return invalid_arg(kFpiKernelName, "lp_flat size = " + std::to_string(lp_count) +
                                           " not divisible by total_S = " + std::to_string(total_S));
  }
  const int64_t batch = lp_count / total_S;
  PYMDP_TRY(check_count(kFpiKernelName, "ll_flat", ll_flat_dev.element_count(), batch * total_ll));
  PYMDP_TRY(check_count(kFpiKernelName, "q_out", q_out_dev->element_count(), batch * total_S));

  // Diagnostic bypass: when PYMDP_FFI_FPI_KERNEL_NOOP=1 is set, short-circuit
  // to an empty kernel with the same grid/block/shmem footprint as the real
  // launch_fpi. Bench delta vs the real kernel at the same fixture shape
  // bounds the XLA-dispatch + cuLaunchKernel + driver-roundtrip overhead
  // share of fixture wall time. Skips the cache build/upload too — those are
  // only there to feed the real kernel, and on cache hit they're already
  // ~free, so isolating launch-only overhead is the cleaner measurement.
  // Static-init read so we don't pay getenv() per call.
  static const bool kFpiNoopMode = []() {
    const char* v = std::getenv("PYMDP_FFI_FPI_KERNEL_NOOP");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (kFpiNoopMode) {
    CUDA_TRY("fpi_cuda noop launch",
             fpi_cuda::launch_fpi_noop(static_cast<int>(batch), static_cast<int>(total_S), stream));
    return FfiError::Success();
  }

  // Pick the kernel variant up-front: small models go through the cmem-
  // parameter-bank path (LDC reads, no per-call H2D for the dispatch table),
  // larger ones use the existing pointer-fed path (LDG reads via device
  // buffers staged on cache miss). The choice is purely a function of F/M;
  // production rollouts (F=3, M=3) all land on the smallmeta path.
  const bool use_smallmeta =
      F <= static_cast<int64_t>(fpi_cuda::kMaxFSmallMeta) && M <= static_cast<int64_t>(fpi_cuda::kMaxMSmallMeta);

  if (!host_cache_hit) {
    // Build per-modality dispatch on host. Validate K range — if any modality
    // has K >= 4 the gate in _fpi.py screwed up; fail loudly rather than
    // produce silent garbage from the kernel's no-op default arm.
    std::vector<int32_t>                       S_host(F);
    std::vector<int32_t>                       lp_offsets_host(F);
    std::vector<fpi_cuda::ModalityDispatchGpu> mods_host(M);
    for (int64_t f = 0; f < F; ++f) {
      S_host[f]          = static_cast<int32_t>(S_span[f]);
      lp_offsets_host[f] = static_cast<int32_t>(lp_offsets[f]);
    }
    for (int64_t m = 0; m < M; ++m) {
      const int64_t dep_start = A_dep_offsets[m];
      const int64_t K         = A_dep_offsets[m + 1] - dep_start;
      if (K < 1 || K > fpi_cuda::kRankMax) {
        return invalid_arg(kFpiKernelName, "FpiCudaDevice handles modality K in [1, " +
                                               std::to_string(fpi_cuda::kRankMax) + "]; modality " + std::to_string(m) +
                                               " has K = " + std::to_string(K));
      }
      fpi_cuda::ModalityDispatchGpu& md = mods_host[m];
      md.K                              = static_cast<int32_t>(K);
      md.ll_off                         = static_cast<int32_t>(ll_offsets[m]);
      for (int64_t i = 0; i < K; ++i) {
        const int64_t d = A_dep_flat[dep_start + i];
        if (d < 0 || d >= F) {
          return invalid_arg(kFpiKernelName, "modality " + std::to_string(m) + " references out-of-range factor");
        }
        md.Ss[i]      = S_host[d];
        md.lp_offs[i] = lp_offsets_host[d];
      }
      for (int64_t i = K; i < fpi_cuda::kRankMax; ++i) {
        md.Ss[i]      = 0;
        md.lp_offs[i] = 0;
      }
    }

    if (use_smallmeta) {
      // Pack the host-side dispatch into the by-value smallmeta struct that
      // we'll hand to launch_fpi_smallmeta as a kernel argument. Pad the
      // tail with zeros so the unused slots are deterministic — the kernel
      // never reads them (loops bound by F / M), but well-defined values
      // make memory-checker tools quieter and prevent stale data from
      // earlier shapes leaking into the cmem image.
      fpi_cuda::FpiSmallMeta& sm = g_fpi_cuda_dev_scratch.smallmeta;
      for (int64_t f = 0; f < F; ++f) {
        sm.S[f]          = S_host[f];
        sm.lp_offsets[f] = lp_offsets_host[f];
      }
      for (int64_t f = F; f < fpi_cuda::kMaxFSmallMeta; ++f) {
        sm.S[f]          = 0;
        sm.lp_offsets[f] = 0;
      }
      for (int64_t m = 0; m < M; ++m) {
        sm.mods[m] = mods_host[m];
      }
      for (int64_t m = M; m < fpi_cuda::kMaxMSmallMeta; ++m) {
        sm.mods[m].K      = 0;
        sm.mods[m].ll_off = 0;
        for (int i = 0; i < fpi_cuda::kRankMax; ++i) {
          sm.mods[m].Ss[i]      = 0;
          sm.mods[m].lp_offs[i] = 0;
        }
      }
      // No device-buffer alloc / H2D in this path — the dispatch travels
      // with the kernel launch as a parameter, served from cmem.
    } else {
      // Resize device-side scratch (no-op when capacity matches) and queue
      // the H2D for the dispatch arrays on the FFI stream.
      CUDA_TRY("fpi_cuda S_dev", g_fpi_cuda_dev_scratch.S_dev.ensure(static_cast<size_t>(F) * sizeof(int32_t)));
      CUDA_TRY("fpi_cuda lp_offsets_dev",
               g_fpi_cuda_dev_scratch.lp_offsets_dev.ensure(static_cast<size_t>(F) * sizeof(int32_t)));
      CUDA_TRY("fpi_cuda mods_dev",
               g_fpi_cuda_dev_scratch.mods_dev.ensure(static_cast<size_t>(M) * sizeof(fpi_cuda::ModalityDispatchGpu)));

      CUDA_TRY("fpi_cuda H2D S",
               cudaMemcpyAsync(g_fpi_cuda_dev_scratch.S_dev.ptr, S_host.data(),
                               static_cast<size_t>(F) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
      CUDA_TRY("fpi_cuda H2D lp_offsets",
               cudaMemcpyAsync(g_fpi_cuda_dev_scratch.lp_offsets_dev.ptr, lp_offsets_host.data(),
                               static_cast<size_t>(F) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
      CUDA_TRY("fpi_cuda H2D mods", cudaMemcpyAsync(g_fpi_cuda_dev_scratch.mods_dev.ptr, mods_host.data(),
                                                    static_cast<size_t>(M) * sizeof(fpi_cuda::ModalityDispatchGpu),
                                                    cudaMemcpyHostToDevice, stream));
    }
    g_fpi_cuda_dev_scratch.sig = sig;
  }

  // Refresh pointer-identity cache record whenever the incoming attr
  // pointers differ from the last record. Covers both the upload-miss path
  // (sig changed → re-uploaded above) and the rare content-hit-with-
  // different-pointers case (sig matched but pointers diverged — possible
  // across executables sharing identical model metadata). When ptr_cache_hit
  // already fired, the record is byte-identical to the spans, so this is a
  // no-op-equivalent set.
  if (!ptr_cache_hit) {
    g_fpi_cuda_dev_scratch.last_S_ptr              = S_span.begin();
    g_fpi_cuda_dev_scratch.last_S_size             = S_span.size();
    g_fpi_cuda_dev_scratch.last_ll_offsets_ptr     = ll_offsets.begin();
    g_fpi_cuda_dev_scratch.last_ll_offsets_size    = ll_offsets.size();
    g_fpi_cuda_dev_scratch.last_lp_offsets_ptr     = lp_offsets.begin();
    g_fpi_cuda_dev_scratch.last_lp_offsets_size    = lp_offsets.size();
    g_fpi_cuda_dev_scratch.last_A_dep_flat_ptr     = A_dep_flat.begin();
    g_fpi_cuda_dev_scratch.last_A_dep_flat_size    = A_dep_flat.size();
    g_fpi_cuda_dev_scratch.last_A_dep_offsets_ptr  = A_dep_offsets.begin();
    g_fpi_cuda_dev_scratch.last_A_dep_offsets_size = A_dep_offsets.size();
  }

  if (use_smallmeta) {
    CUDA_TRY("fpi_cuda launch (smallmeta)",
             fpi_cuda::launch_fpi_smallmeta(ll_flat_dev.typed_data(), lp_flat_dev.typed_data(), q_out_dev->typed_data(),
                                            static_cast<int>(batch), static_cast<int>(F), static_cast<int>(M),
                                            static_cast<int>(total_ll), static_cast<int>(total_S), num_iter,
                                            g_fpi_cuda_dev_scratch.smallmeta, stream));
  } else {
    CUDA_TRY("fpi_cuda launch",
             fpi_cuda::launch_fpi(ll_flat_dev.typed_data(), lp_flat_dev.typed_data(), q_out_dev->typed_data(),
                                  static_cast<int>(batch), static_cast<int>(F), static_cast<int>(M),
                                  static_cast<int>(total_ll), static_cast<int>(total_S), num_iter,
                                  g_fpi_cuda_dev_scratch.S_dev.as<const int32_t>(),
                                  g_fpi_cuda_dev_scratch.lp_offsets_dev.as<const int32_t>(),
                                  g_fpi_cuda_dev_scratch.mods_dev.as<const fpi_cuda::ModalityDispatchGpu>(), stream));
  }

  return FfiError::Success();
}

#endif  // PYMDP_FFI_HAS_CUDA

}  // namespace pymdp_ffi
