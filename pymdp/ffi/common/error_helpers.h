// Small helpers shared between the pymdp FFI kernels that depend on xla::ffi
// types. Kept separate from kernel_primitives.h so the latter stays usable
// without the XLA FFI headers.

#pragma once

#include <cstdint>
#include <string>

#include "xla/ffi/api/ffi.h"

namespace pymdp_ffi {

namespace ffi = ::xla::ffi;

using FfiError     = ffi::Error;
using FfiF32Buf    = ffi::Buffer<ffi::F32>;
using FfiS32Buf    = ffi::Buffer<ffi::S32>;
using FfiF32Out    = ffi::Result<FfiF32Buf>;
using FfiInt64Span = ffi::Span<const int64_t>;

// Build a kInvalidArgument FFI error with a kernel-name prefix. The prefix
// makes the error message self-identifying when surfaced through JAX.
inline FfiError invalid_arg(const char* prefix, const std::string& msg) {
  return FfiError(ffi::ErrorCode::kInvalidArgument, std::string(prefix) + ": " + msg);
}

inline FfiError check_span_size(const char* prefix, const char* name, int64_t actual, int64_t expected) {
  if (actual != expected) {
    return invalid_arg(prefix, std::string(name) + " size = " + std::to_string(actual) + ", expected " +
                                   std::to_string(expected));
  }
  return FfiError::Success();
}

inline FfiError check_count(const char* prefix, const char* name, int64_t actual, int64_t expected) {
  if (actual != expected) {
    return invalid_arg(prefix, std::string(name) + " element_count = " + std::to_string(actual) + ", expected " +
                                   std::to_string(expected));
  }
  return FfiError::Success();
}

inline FfiError check_monotonic(const char* prefix, const char* name, int64_t prev, int64_t curr) {
  if (curr < prev) {
    return invalid_arg(prefix, std::string(name) + " must be monotonic");
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi

// Early-return on ffi::Error failure. `_pymdp_err_` avoids shadowing a
// caller-side `err`. cuda_err / cublas_err in common/cuda_memory.h
// delegate through this macro.
#define PYMDP_TRY(expr)                                                                                                \
  do {                                                                                                                 \
    if (auto _pymdp_err_ = (expr); _pymdp_err_.failure()) return _pymdp_err_;                                          \
  } while (0)

// Branch hints. C++17 lacks [[likely]] / [[unlikely]]; gcc/clang both accept
// __builtin_expect. Use sparingly: dynamic branch predictors already handle
// most patterns correctly. Right targets are >99%-skewed branches that the
// compiler can't infer statically (cache miss after warmup, validation failure).
#define PYMDP_LIKELY(x) __builtin_expect(static_cast<bool>(x), true)
#define PYMDP_UNLIKELY(x) __builtin_expect(static_cast<bool>(x), false)