// ABI declarations consumed by xla_register.cc — keep byte-identical with
// FpiCpu (fpi_cpu_runner.cc), FpiCudaHost (fpi_cuda_host_shim.cc),
// FpiCudaDevice (fpi_cuda_runtime.cc), and pymdp/ffi/_fpi.py buffer /
// attribute packing.

#pragma once

#include <cstdint>

#ifdef PYMDP_FFI_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "common/error_helpers.h"  // FfiError / FfiF32Buf / FfiF32Out / FfiInt64Span

namespace pymdp_ffi {

FfiError FpiCpu(FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S, FfiInt64Span ll_offsets,
                FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter);

#ifdef PYMDP_FFI_HAS_CUDA
// platform="CUDA" target: D2H ll/lp into thread_local host scratch, run the
// validated CPU kernel, H2D q_out into JAX's device output buffer. Same
// ABI as FpiCpu plus the CUDA stream (PlatformStream context).
FfiError FpiCudaHost(cudaStream_t stream, FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S,
                     FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                     FfiInt64Span A_dep_offsets, int32_t num_iter);

// platform="CUDA" target: native CUDA FPI. Single kernel runs all `num_iter`
// iterations internally — no D2H/H2D, no host-side stream sync. Restricted
// to modalities with K in [1, 3]; the host-side gate in pymdp/ffi/_fpi.py
// dispatches K>=4 cases to FpiCudaHost (shim) instead.
FfiError FpiCudaDevice(cudaStream_t stream, FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S,
                       FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                       FfiInt64Span A_dep_offsets, int32_t num_iter);
#endif

}  // namespace pymdp_ffi
