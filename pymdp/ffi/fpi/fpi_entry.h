// Attribute parse + validate for the FPI FFI ABI.
//
// Mirrors neg_efe/neg_efe_entry.h's role: the host-side validator that every
// FPI entry (FpiCpu, FpiCudaHost, FpiCudaDevice) runs before dispatching to
// its specific pipeline. Pure attr work — no buffer reads — so cheap to run
// on every call.

#pragma once

#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"

namespace pymdp_ffi {

inline constexpr const char* kFpiKernelName = "fpi_ffi";

// Validate attr-side invariants: F/M/num_iter positive, span sizes match
// F+1 / M+1, S[f] positive, and lp/ll/A_dep offsets monotonic. Pure attr
// work — no buffer reads.
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

}  // namespace pymdp_ffi
