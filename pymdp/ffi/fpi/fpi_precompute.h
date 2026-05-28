// Per-modality CPU dispatch metadata + build path for the FPI kernel.
//
// ModalityDispatch is kept distinct from neg_efe's `ModalityMeta` because
// their fields are tightly bound to per-kernel precompute pipelines that
// don't overlap. The `FpiModalityMeta` typedef provides a seam that future
// struct unification can land against without touching call sites.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "common/modality_dispatch.h"
#include "common/scratch_arena.h"
#include "fpi/fpi_entry.h"

namespace pymdp_ffi {

// Per-modality dispatch info resolved once in run_fpi_kernel_host so the
// num_iter * M hot loop can jump straight to modality_K{1,2,3} or
// modality_Kn (K>=4).
//
// For K<=3, Ss[0..2] + lp_offs[0..2] hold the per-position state size and
// log_q offset (the K=1/K=2/K=3 paths read these directly).
//
// For K>=4, Ss[i] / lp_offs[i] (i in [0, K)) carry the same info for the
// generic modality_Kn path; the (tail, suf_offs, f_offs) triple is the
// hoisted per-iter setup.
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
  // K>=4 only: precomputed by build_modality_dispatch, read by modality_Kn
  // each iter:
  //   tail[k]     = prod_{i>k} Ss[i],  tail[K-1] = 1
  //   suf_offs[k] = offset into suffix_buf for suffix_q[k]
  //   f_offs[k]   = offset into fchain_buf for F_k (f_offs[0]/f_offs[1] = 0)
  // Hoisted out of modality_Kn so the num_iter * M call site doesn't redo
  // the K-pass computation each invocation. Ignored for K<=3 hot paths.
  // Stay int64_t — products of state sizes can overflow int32_t at the
  // rank-8 ABI cap.
  std::array<int64_t, kMaxFfiDependencyRank> tail;
  std::array<int64_t, kMaxFfiDependencyRank> suf_offs;
  std::array<int64_t, kMaxFfiDependencyRank> f_offs;
};

// Seam for future unification with neg_efe's ModalityMeta; the two structs
// share (K, Ss, lp_offs) but FPI adds K>=4 F-chain offsets.
using FpiModalityMeta = ModalityDispatch;

// Build the per-modality dispatch table. Validates each modality's dep rank
// and factor refs, populates the K hot-path fields (Ss, lp_offs, ll_off), and
// for K>=4 precomputes the (tail, suf_offs, f_offs) triples used by
// modality_Kn so the num_iter*M hot loop doesn't redo the scalar setup.
// Reports the worst-case scratch sizes (max_t01 / max_prefix / max_suffix) via
// out-params for the caller's scratch sizing.
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
    const std::string mod_label = "modality " + std::to_string(m) + " A_dep";
    PYMDP_TRY(validate_dep_rank_and_factors(kFpiKernelName, mod_label.c_str(), F, K, A_dep_flat.begin() + dep_start));
    for (int64_t i = 0; i < K; ++i) {
      const int64_t d = A_dep_flat[dep_start + i];
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

}  // namespace pymdp_ffi
