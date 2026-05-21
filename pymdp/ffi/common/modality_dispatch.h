// Shared modality / factor dependency primitives for the neg-EFE and FPI
// kernels: the dep-rank constant and the dep-validation helper that both
// kernels' host-side build paths would otherwise duplicate.
//
// The per-kernel `ModalityDispatch` (fpi_precompute.h) and `ModalityMeta`
// (neg_efe_precompute.h) structs stay where they are. Unifying them would
// require forcing fields from both kernels' precompute pipelines (FPI's
// K>=4 F-chain offsets vs neg-EFE's PrecomputedHA / TransposedB offsets)
// into a single struct, adding more code than it removes. fpi/fpi_precompute.h
// carries a typedef seam (`using FpiModalityMeta = ModalityDispatch`) so a
// future unification can land without touching call sites.

#pragma once

#include <cstdint>
#include <string>

#include "common/error_helpers.h"

namespace pymdp_ffi {

// Maximum modality / factor dependency rank shared by the neg-EFE and FPI
// FFI paths. `MAX_FFI_DEP_RANK` in pymdp/ffi/_core.py must match, and
// neg_efe_cuda_kernels.h duplicates the numeric cap with a static_assert
// (nvcc 10.2 cannot include XLA FFI api.h transitively, so it cannot include
// this header either; the duplicate-by-static_assert is intentional).
inline constexpr int kMaxFfiDependencyRank = 8;

// Validate that a modality / factor dependency view has rank in [1, MaxRank]
// and that every factor index lies in [0, F). The body matches the build-
// path checks both kernels run today (fpi_precompute.h:build_modality_dispatch
// and neg_efe_layout.h:validate_layout).
//
// `label` is the human-readable error category ("A_dep", "B_dep", or
// "modality m"); `kernel_name` is the FfiError owner ("efe_ffi", "fpi_ffi").
inline FfiError validate_dep_rank_and_factors(const char* kernel_name, const char* label, int64_t F, int64_t rank,
                                              const int64_t* factors, int max_rank = kMaxFfiDependencyRank) {
  if (rank < 1 || rank > max_rank) {
    return invalid_arg(kernel_name, std::string(label) + " rank must be in [1, " + std::to_string(max_rank) + "]");
  }
  for (int64_t i = 0; i < rank; ++i) {
    const int64_t fi = factors[i];
    if (fi < 0 || fi >= F) {
      return invalid_arg(kernel_name, std::string(label) + " references out-of-range factor");
    }
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi
