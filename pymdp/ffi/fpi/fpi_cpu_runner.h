// Shared host-side FPI driver. Both FpiCpu (platform="cpu", host buffers)
// and FpiCudaHost (platform="CUDA", device buffers aliased / D2H'd into
// host scratch) funnel through `run_fpi_kernel_host` so the validated
// K=1/2/3 hot paths + K>=4 forward-chain generic path + batch OMP path are
// written exactly once.

#pragma once

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"

namespace pymdp_ffi {

FfiError run_fpi_kernel_host(const float* ll_flat, int64_t ll_count, const float* lp_flat, int64_t lp_count,
                             float* q_out, int64_t q_count, FfiInt64Span S, FfiInt64Span ll_offsets,
                             FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets,
                             int32_t num_iter);

}  // namespace pymdp_ffi
