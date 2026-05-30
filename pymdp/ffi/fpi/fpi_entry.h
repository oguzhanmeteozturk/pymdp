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
  PYMDP_TRY(check_span_size(kFpiKernelName, "A_dep_offsets", static_cast<int64_t>(A_dep_offsets.size()), M + 1));
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

// Validate the runtime (batch-dependent) buffer shapes against the static
// per-iteration sizes and return the inferred batch via `*batch_out`. lp_flat
// is [batch, total_S]; ll_flat is [batch, total_ll]; q_out is [batch, total_S].
// The element counts come from the JAX runtime (the vmap batch dim), not the
// static attrs, so the same model metadata can be called at different batch
// sizes. Shared by the CPU (FpiCpu / FpiCudaHost) and native-CUDA paths, which
// differ only in how they obtain the counts (explicit args vs Buffer counts).
inline FfiError validate_fpi_batch_shapes(int64_t lp_count, int64_t ll_count, int64_t q_count, int64_t total_S,
                                          int64_t total_ll, int64_t* batch_out) {
  if (total_S <= 0 || lp_count <= 0 || lp_count % total_S != 0) {
    return invalid_arg(kFpiKernelName, "lp_flat size = " + std::to_string(lp_count) +
                                           " not divisible by total_S = " + std::to_string(total_S));
  }
  const int64_t batch = lp_count / total_S;
  PYMDP_TRY(check_count(kFpiKernelName, "ll_flat", ll_count, batch * total_ll));
  PYMDP_TRY(check_count(kFpiKernelName, "q_out", q_count, batch * total_S));
  *batch_out = batch;
  return FfiError::Success();
}

// Reject a duplicate factor within one modality's A_dependencies. The hot-loop
// kernels mark each factor's q / log_q slices __restrict__, so an aliased
// (duplicated) factor would be silent UB that corrupts the update. Python's
// can_handle_fpi rejects duplicates up front; this is the C++ ABI-boundary
// re-check, shared by the CPU dispatch build (build_modality_dispatch) and the
// CUDA cache refresh. `lp_offs[0..count)` are the offsets already accepted for
// modality `m`; `lp_off` is the candidate being added.
inline FfiError check_distinct_modality_factor(int64_t m, const int32_t* lp_offs, int64_t count, int32_t lp_off) {
  for (int64_t j = 0; j < count; ++j) {
    if (lp_offs[j] == lp_off) {
      return invalid_arg(kFpiKernelName, "modality " + std::to_string(m) + " has duplicate factor in A_dependencies");
    }
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi
