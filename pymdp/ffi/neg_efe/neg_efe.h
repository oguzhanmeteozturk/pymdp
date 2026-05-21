// ABI declarations consumed by xla_register.cc — keep byte-identical with
// NegEfeCpu / NegEfeCudaHost / NegEfeCudaDev and pymdp/ffi/_efe.py packing.

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>

#include "xla/ffi/api/ffi.h"

#ifdef PYMDP_FFI_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "common/error_helpers.h"  // FfiError / FfiF32Buf / FfiS32Buf / FfiF32Out / FfiInt64Span

namespace pymdp_ffi {

#ifdef PYMDP_FFI_HAS_CUDA
// Per-jit-instance state for the CUDA-platform NegEfe target. Lifetime is
// tied to the compiled XLA executable: BindInstantiate runs once per HLO
// custom-call instance, and the State<NegEfeState>* pointer is stable across
// every Execute call on that instance. Owns the four model-parameter caches
// (tree / A / B / linear), the per-call scratch, and the cuBLAS handle —
// see NegEfeContext in neg_efe_cuda_context.h. pImpl keeps the FFI header clean
// of CUDA + per-cache implementation details.
struct NegEfeContext;

struct NegEfeState {
  static ffi::TypeId id;

  std::unique_ptr<NegEfeContext> ctx;

  NegEfeState();
  ~NegEfeState();
};

ffi::ErrorOr<std::unique_ptr<NegEfeState>> NegEfeCudaInstantiate();

// CUDA-platform Execute. Bind chain prepends Ctx<PlatformStream<cudaStream_t>>
// and Ctx<State<NegEfeState>> ahead of the shared NegEfe buffer + attribute ABI;
// see xla_register.cc PYMDP_NEG_EFE_ARGS_AND_ATTRS.
FfiError NegEfeCudaDev(cudaStream_t stream, NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A,
                       FfiF32Buf B, FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                       FfiF32Out out, FfiInt64Span S, FfiInt64Span O, FfiInt64Span U, FfiInt64Span qs_offsets,
                       FfiInt64Span A_offsets, FfiInt64Span B_offsets, FfiInt64Span C_offsets, FfiInt64Span I_offsets,
                       FfiInt64Span I_depths, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets,
                       FfiInt64Span B_dep_flat, FfiInt64Span B_dep_offsets, int32_t flags);
#endif  // PYMDP_FFI_HAS_CUDA

// Shared NegEfe FFI ABI (CPU / CUDA host / CUDA device): 9 input buffers +
// 1 output buffer + 13 int64 span attributes + int32 flags.
FfiError NegEfeCpu(FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B, FfiF32Buf C, FfiF32Buf I,
                   FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon, FfiF32Out out, FfiInt64Span S,
                   FfiInt64Span O, FfiInt64Span U, FfiInt64Span qs_offsets, FfiInt64Span A_offsets,
                   FfiInt64Span B_offsets, FfiInt64Span C_offsets, FfiInt64Span I_offsets, FfiInt64Span I_depths,
                   FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, FfiInt64Span B_dep_flat,
                   FfiInt64Span B_dep_offsets, int32_t flags);

#ifdef PYMDP_FFI_HAS_CUDA
// CUDA port, host-buffer variant. Registered on platform="cpu" so XLA hands
// us host buffers; the kernel internally manages cudaMalloc / launch /
// cudaMemcpy and writes the result back to host. Used wherever JAX runs on
// CPU but the CUDA FFI is selected — primarily hosts where jaxlib is
// CPU-only, and x86 with `JAX_PLATFORMS=cpu`. Multi-stage handler;
// per-JIT NegEfeState owns the caches, same as NegEfeCudaDev.
FfiError NegEfeCudaHost(NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B,
                        FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                        FfiF32Out out, FfiInt64Span S, FfiInt64Span O, FfiInt64Span U, FfiInt64Span qs_offsets,
                        FfiInt64Span A_offsets, FfiInt64Span B_offsets, FfiInt64Span C_offsets, FfiInt64Span I_offsets,
                        FfiInt64Span I_depths, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets,
                        FfiInt64Span B_dep_flat, FfiInt64Span B_dep_offsets, int32_t flags);
#endif

}  // namespace pymdp_ffi
