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
#include <cstring>
#include <string>
#include <vector>

#include <omp.h>

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
                        float* __restrict__ scratch_t01, float* /*scratch_t12 unused*/) {
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
                        float* __restrict__ suffix_buf, float* /*tmp_unused*/) {
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
// `__restrict__` on each member guarantees no-alias across all members —
// every member maps to its own std::vector allocation in FpiScratch.
// GCC/clang extension; supported by both compilers the FFI is built with.
struct FpiScratchPtrs {
  float* __restrict__ log_q;
  float* __restrict__ log_q_prev;
  float* __restrict__ q;
  float* __restrict__ t01;
  float* __restrict__ prefix;
  float* __restrict__ suffix;
  float* __restrict__ tmp;
};

struct FpiScratch {
  std::vector<float>            log_q;
  std::vector<float>            log_q_prev;  // convergence-check snapshot of prior iter's log_q
  std::vector<float>            q;
  std::vector<float>            t01;     // K=3 shared prefix (still used for marg0 sgemv)
  std::vector<float>            prefix;  // K>=4 prefix_q (incrementally extended)
  std::vector<float>            suffix;  // K>=4 suffix_q tensors back-to-back
  std::vector<float>            tmp;     // K>=4 step-a output (also marg_0 holding spot)
  std::vector<ModalityDispatch> mods;

  void ensure_dispatch(int64_t M) { ensure_at_least(mods, M); }
  void ensure_buffers(int64_t total_S, int64_t max_t01, int64_t max_prefix, int64_t max_suffix, int64_t max_tmp) {
    ensure_at_least(log_q, total_S);
    ensure_at_least(log_q_prev, total_S);
    ensure_at_least(q, total_S);
    ensure_at_least(t01, max_t01);
    ensure_at_least(prefix, max_prefix);
    ensure_at_least(suffix, max_suffix);
    ensure_at_least(tmp, max_tmp);
  }

  FpiScratchPtrs as_ptrs() {
    return {log_q.data(), log_q_prev.data(), q.data(), t01.data(), prefix.data(), suffix.data(), tmp.data()};
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
  float* __restrict__ const scratch_tmp        = sc.tmp;

  // log_q starts at zero, matching the JAX scan reference in run_factorized_fpi.
  std::memset(scratch_log_q, 0, total_S * sizeof(float));

  for (int64_t it = 0; it < num_iter; ++it) {
    std::memcpy(scratch_log_q_prev, scratch_log_q, total_S * sizeof(float));

    // q[f] = softmax(log_q[f])
    softmax_per_factor(F, S, lp_offsets, scratch_log_q, scratch_q);

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
                    scratch_log_q + md.lp_offs[2], scratch_t01, /*scratch_t12=*/nullptr);
      } else {  // md.K in [4, kMaxFfiDependencyRank]
        std::array<const float*, kMaxFfiDependencyRank> qs_ptrs{};
        std::array<float*, kMaxFfiDependencyRank>       log_q_outs{};
        for (int i = 0; i < md.K; ++i) {
          qs_ptrs[i]    = scratch_q + md.lp_offs[i];
          log_q_outs[i] = scratch_log_q + md.lp_offs[i];
        }
        modality_Kn(md.K, ll_m, md.Ss.data(), md.tail.data(), md.suf_offs.data(), md.f_offs.data(), qs_ptrs.data(),
                    log_q_outs.data(), scratch_prefix, scratch_suffix, scratch_tmp);
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

// =============================================================================
// ABI entry point
// =============================================================================

FfiError FpiCpu(FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S, FfiInt64Span ll_offsets,
                FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter) {
  const int64_t F = static_cast<int64_t>(S.size());
  const int64_t M = static_cast<int64_t>(A_dep_offsets.size()) - 1;

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

  const int64_t total_S  = lp_offsets[F];
  const int64_t total_ll = ll_offsets[M];
  const int64_t lp_total = lp_flat.element_count();

  if (total_S <= 0 || lp_total <= 0 || lp_total % total_S != 0) {
    return invalid_arg(kFpiKernelName, "lp_flat size = " + std::to_string(lp_total) +
                                           " not divisible by total_S = " + std::to_string(total_S));
  }
  const int64_t batch = lp_total / total_S;

  PYMDP_TRY(check_count(kFpiKernelName, "ll_flat", ll_flat.element_count(), batch * total_ll));
  PYMDP_TRY(check_count(kFpiKernelName, "q_out", q_out->element_count(), batch * total_S));

  // Caller-validated invariants — restate so the optimizer can drop zero-trip
  // guards on the per-batch / per-factor / per-modality loops below.
  if (F <= 0 || M <= 0 || batch <= 0 || total_S <= 0 || num_iter <= 0) __builtin_unreachable();

  // Decode dependencies once for scratch sizing and the hot loop.
  //   K=3 hot path: scratch_t01 holds the shared prefix for marg0's sgemv.
  //   K>=4 generic path: scratch_prefix/suffix hold the F-chain and suffix_q
  //   tensors (sized to the worst-case modality).
  // All scratch is shared across modalities — modalities run sequentially within
  // a single batch element, so reuse is safe.
  //
  // mods table + scratch buffers backed by thread_local g_fpi_scratch so
  // repeated FPI calls of the same shape allocate zero times after warm-up.
  FpiScratch& sc = g_fpi_scratch;
  sc.ensure_dispatch(M);
  std::vector<ModalityDispatch>& mods       = sc.mods;
  int64_t                        max_t01    = 0;
  int64_t                        max_prefix = 0, max_suffix = 0, max_tmp = 0;
  for (int64_t m = 0; m < M; ++m) {
    const int64_t     dep_start = A_dep_offsets[m];
    const int64_t     K         = A_dep_offsets[m + 1] - dep_start;
    ModalityDispatch& md        = mods[m];
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
      // Distinct factors per modality. The hot-loop kernels assume q[deps[i]]
      // and q[deps[j]] (and the matching log_q slices) don't alias, which
      // lets us mark them __restrict__ for vectorization. Duplicates would
      // be silently UB. Python's can_handle_fpi rejects them up front; this
      // is the C++ re-check at the ABI boundary.
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
      max_t01 = std::max<int64_t>(max_t01, static_cast<int64_t>(md.Ss[0]) * md.Ss[1]);
    }
    if (K >= 4) {
      // Precompute md.tail / md.suf_offs / md.f_offs once per call so
      // modality_Kn (which runs num_iter times per K>=4 modality) can skip
      // the K-pass scalar setup on every invocation. Definitions:
      //   tail[k]     = prod_{i>k} Ss[i];  tail[K-1] = 1
      //   suf_offs[k] = offset within suffix_buf for suffix_q[k] (size tail[k])
      //   f_offs[k]   = offset within fchain_buf for F_k (size Ss[k]*tail[k]);
      //                 f_offs[0]/f_offs[1] are 0 (F_0 aliases ll directly)
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

      // F-chain storage (FpiScratch::prefix slot): F_1..F_{K-1} back-to-back,
      //   total = f_offs[K-1] + Ss[K-1] * tail[K-1].
      // Suffix chain (FpiScratch::suffix slot): suffix_q[k] back-to-back,
      //   total = suf_offs[0] + tail[0].
      // tmp_buf is unused on the K>=4 path (sgemv_rm_f32_add writes straight
      // into log_q). Keep max_tmp at zero so the scratch isn't grown.
      const int64_t fchain_total = md.f_offs[K - 1] + md.Ss[K - 1] * md.tail[K - 1];
      const int64_t suffix_total = md.suf_offs[0] + md.tail[0];
      max_prefix                 = std::max<int64_t>(max_prefix, fchain_total);
      max_suffix                 = std::max<int64_t>(max_suffix, suffix_total);
    }
  }

  KernelTimer timer([&](double us) {
    std::fprintf(stderr, "[fpi] batch=%lld F=%lld M=%lld iter=%d total_S=%lld | total=%6.2f us\n",
                 static_cast<long long>(batch), static_cast<long long>(F), static_cast<long long>(M),
                 static_cast<int>(num_iter), static_cast<long long>(total_S), us);
  });

  const int64_t ni                  = num_iter;
  const bool    fire_batch_parallel = should_parallelize_fpi_batch(batch, ni, total_ll);
  // Serial vs parallel paths are split (rather than gated by `omp parallel
  // if(...)`) so the serial path never enters the OpenMP runtime. An
  // `if(false)` parallel region still creates a 1-thread team and does
  // libgomp/libomp bookkeeping — measurable on the batch=1 path where
  // per-call kernel time is ~30-200us. Splitting here drops that to a plain
  // function call.
  if (!fire_batch_parallel) {
    FpiScratch& sc_thread = g_fpi_scratch;
    sc_thread.ensure_buffers(total_S, max_t01, max_prefix, max_suffix, max_tmp);
    const FpiScratchPtrs sc_ptrs = sc_thread.as_ptrs();
    for (int64_t b = 0; b < batch; ++b) {
      fpi_one_batch(ni, F, M, total_S, S.begin(), lp_offsets.begin(), lp_flat.typed_data() + b * total_S,
                    ll_flat.typed_data() + b * total_ll, mods.data(), q_out->typed_data() + b * total_S, sc_ptrs);
    }
  } else {
#pragma omp parallel num_threads(omp_team_size_for_work_units(batch))
    {
      FpiScratch& sc_thread = g_fpi_scratch;
      sc_thread.ensure_buffers(total_S, max_t01, max_prefix, max_suffix, max_tmp);
      const FpiScratchPtrs sc_ptrs = sc_thread.as_ptrs();
#pragma omp for schedule(static)
      for (int64_t b = 0; b < batch; ++b) {
        fpi_one_batch(ni, F, M, total_S, S.begin(), lp_offsets.begin(), lp_flat.typed_data() + b * total_S,
                      ll_flat.typed_data() + b * total_ll, mods.data(), q_out->typed_data() + b * total_S, sc_ptrs);
      }
    }
  }

  return FfiError::Success();
}

}  // namespace pymdp_ffi
