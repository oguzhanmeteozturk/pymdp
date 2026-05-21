// CPU hot-math primitives + single-batch driver for the FPI kernel.
//
// Header-inline by cross-cutting rule #3: these are the per-modality and
// per-batch hot loops, so cross-TU inlining via LTO is load-bearing. Stays
// header-only so fpi_cpu_runner.cc and any future ABI shim share one
// definition.
//
// Pulls softmax_per_factor / softmax_per_factor_snapshot / max_abs_diff /
// kFpiConvergenceTol from common/convergence_helpers.h and the sgemv /
// axpy primitives from common/kernel_primitives.h.

#pragma once

#include <array>
#include <cstdint>
#include <cstring>

#include "common/convergence_helpers.h"
#include "common/kernel_primitives.h"
#include "common/modality_dispatch.h"
#include "fpi/fpi_precompute.h"

namespace pymdp_ffi {

// =============================================================================
// Per-modality inner ops
// =============================================================================

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
// dedup check in build_modality_dispatch), log_q_d0/log_q_d1 are
// non-overlapping slices of scratch_log_q. Restrict lets the compiler keep
// q1[s1] and qs0 in registers across the log_q_d1 RMW.
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
// Algorithm (asymptotically the same cost as JAX/opt_einsum's CSE-ed
// marginals — ~K * prod(S) work per modality per iter):
//   1. Build suffix_q chain (reverse): K-2 outer-product passes.
//   2. Build F_k chain (forward): F_0 = ll; F_{k+1} = axpy_fold_leading over
//      F_k's leading axis weighted by q_k.
//   3. Per-k: sgemv_rm_f32_add into log_q_outs[k] using F_k as the matrix and
//      suffix_q[k] as the vector.
//
// Buffers must be sized for the worst-case modality (handled by the caller
// via the max_prefix / max_suffix from build_modality_dispatch).
inline void modality_Kn(int K, const float* __restrict__ ll_m, const int32_t* __restrict__ Ss,
                        const int64_t* __restrict__ tail, const int64_t* __restrict__ suf_offs,
                        const int64_t* __restrict__ f_offs, const float* const* __restrict__ qs,
                        float* const* __restrict__ log_q_outs, float* __restrict__ fchain_buf,
                        float* __restrict__ suffix_buf) {
  // Caller-side precondition: K in [4, kMaxFfiDependencyRank]. The dispatch
  // in fpi_one_batch gates this branch on `md.K >= 4` and
  // build_modality_dispatch validates `K <= kMaxFfiDependencyRank` before
  // populating the dispatch table. `__builtin_unreachable` lets the optimizer
  // eliminate the guard entirely AND silences clang-analyzer-security.
  // ArrayBound on the `tail[K - 1]` reads below.
  if (K < 1 || K > kMaxFfiDependencyRank) __builtin_unreachable();

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
    for (int64_t i = 0; i < qsz; ++i) {
      const float qi  = qk1[i];
      float*      row = dst + i * prev_sz;
      for (int64_t j = 0; j < prev_sz; ++j) row[j] = qi * prev[j];
    }
  }

  // ---- 2. Build F_k chain (forward): F_{k+1}[s_{k+1}..] = sum_{s_k} F_k[s_k..] * q_k[s_k].
  for (int k = 0; k < K - 1; ++k) {
    const float*  fk_in  = (k == 0) ? ll_m : (fchain_buf + f_offs[k]);
    float*        fk_out = fchain_buf + f_offs[k + 1];
    const int64_t lead   = Ss[k];
    const int64_t inner  = tail[k];
    axpy_fold_leading(lead, inner, fk_in, qs[k], fk_out);
  }

  // ---- 3. Per-k marginal accumulation.
  for (int k = 0; k < K; ++k) {
    const float*  fk    = (k == 0) ? ll_m : (fchain_buf + f_offs[k]);
    const int64_t Sk    = Ss[k];
    const int64_t inner = tail[k];
    if (inner == 1) {
      // suffix_q[K-1] is scalar 1; marg_{K-1}[s_k] = F_{K-1}[s_k]. Add directly.
      float* out = log_q_outs[k];
      for (int64_t s = 0; s < Sk; ++s) out[s] += fk[s];
    } else {
      sgemv_rm_f32_add(Sk, inner, fk, inner, suffix_buf + suf_offs[k], log_q_outs[k]);
    }
  }
}

// =============================================================================
// Per-batch types & driver
// =============================================================================

// Bundle of distinct compute-scratch buffer pointers handed to fpi_one_batch.
// `__restrict__` on each member guarantees no-alias across all members —
// every member maps to its own backing buffer in the caller's FpiScratch.
struct FpiScratchPtrs {
  float* __restrict__ log_q;
  float* __restrict__ log_q_prev;
  float* __restrict__ q;
  float* __restrict__ t01;
  float* __restrict__ prefix;
  float* __restrict__ suffix;
};

// Single-batch FPI: runs num_iter iterations and writes the final softmax to
// q_out (flat, layout matches lp_offsets).
inline void fpi_one_batch(int64_t num_iter, int64_t F, int64_t M, int64_t total_S, const int64_t* __restrict__ S,
                          const int64_t* __restrict__ lp_offsets, const float* __restrict__ lp_flat,
                          const float* __restrict__ ll_flat, const ModalityDispatch* __restrict__ mods,
                          float* __restrict__ q_out, const FpiScratchPtrs& sc) {
  // Caller-validated invariants — restate so the optimizer can drop zero-trip
  // guards on per-F / per-M / per-total_S loops in this body.
  if (F <= 0 || M <= 0 || total_S <= 0 || num_iter <= 0) __builtin_unreachable();

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
    // convergence check below. The snapshot is folded into the softmax pass.
    softmax_per_factor_snapshot(F, S, lp_offsets, scratch_log_q, scratch_q, scratch_log_q_prev);

    // Reset accumulator: log_q[f] = log_prior[f]
    std::memcpy(scratch_log_q, lp_flat, total_S * sizeof(float));

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

}  // namespace pymdp_ffi
