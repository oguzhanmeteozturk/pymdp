// Private cross-TU contract between fpi_cuda_cache.cc and
// fpi_cuda_runtime.cc. Not part of the FPI public ABI; runtime calls
// `refresh_fpi_cuda_cache` to ensure the thread_local device-side
// dispatch state matches the incoming attr spans, then reads the
// `FpiCudaDeviceScratch` it returns to drive the launch.

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "fpi/fpi_cuda_context.h"

namespace pymdp_ffi {

// Refresh the thread_local FpiCudaDevice cache for `spans` / (F, M).
// On miss: validates the attrs, builds the host dispatch table, either
// packs `smallmeta` or H2D-uploads `S_dev`/`lp_offsets_dev`/`mods_dev`,
// stamps `sig` + the pointer-identity record, and returns a pointer to
// the now-fresh g_fpi_cuda_dev_scratch. On hit: skips everything and
// returns the existing record.
//
// `num_iter` is part of the runtime guard (positive) but NOT part of the
// cache key (it doesn't affect the dispatch table contents), so this
// helper validates it on cache hits too.
//
// `use_smallmeta_out` reports which launch variant the runtime should
// pick — purely a function of F / M, returned alongside the cache record
// to avoid the runtime re-deriving it.
FfiError refresh_fpi_cuda_cache(FfiInt64Span S_span, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets,
                                FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter, int64_t F,
                                int64_t M, cudaStream_t stream, FpiCudaDeviceScratch** scratch_out,
                                bool* use_smallmeta_out);

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
