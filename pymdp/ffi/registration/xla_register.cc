// XLA FFI handler registration and ctypes-visible trampolines for the pymdp
// FFI kernels (neg-EFE and FPI).
//
// ABI boundary: each Bind chain matches neg_efe.h / fpi.h and the Python
// wrappers in pymdp/ffi/_efe.py, pymdp/ffi/_fpi.py (buffers + attrs).

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#ifdef PYMDP_FFI_HAS_CUDA
#include <cuda_runtime.h>
#endif

#include "neg_efe/neg_efe.h"
#include "fpi/fpi.h"
#include "common/kernel_primitives.h"
#include "common/modality_dispatch.h"  // kMaxFfiDependencyRank — for the cross-ABI cap assert below

namespace ffi = ::xla::ffi;

using ::pymdp_ffi::FfiF32Buf;
using ::pymdp_ffi::FfiInt64Span;
using ::pymdp_ffi::FfiS32Buf;

static_assert(::pymdp_ffi::kMaxFfiDependencyRank == 8);

// Shared Bind-chain tail for all NegEfe targets. Expands to 9 input buffers
// (int32 policy_matrix + 8xf32) + 1xf32 output + 13 Span<int64_t> attrs +
// int32 flags — see neg_efe.h for the full ABI.
#define PYMDP_NEG_EFE_ARGS_AND_ATTRS()                                                                                 \
  .Arg<FfiS32Buf>()     /* policy_matrix */                                                                            \
      .Arg<FfiF32Buf>() /* qs_init */                                                                                  \
      .Arg<FfiF32Buf>() /* A */                                                                                        \
      .Arg<FfiF32Buf>() /* B */                                                                                        \
      .Arg<FfiF32Buf>() /* C */                                                                                        \
      .Arg<FfiF32Buf>() /* I */                                                                                        \
      .Arg<FfiF32Buf>() /* pA — empty when use_param_info_gain off */                                                  \
      .Arg<FfiF32Buf>() /* pB — empty when use_param_info_gain off */                                                  \
      .Arg<FfiF32Buf>() /* inductive_epsilon (scalar or [Bn] under broadcast_all) */                                   \
      .Ret<FfiF32Buf>() /* neg_efe_out */                                                                              \
      .Attr<FfiInt64Span>("S")                                                                                         \
      .Attr<FfiInt64Span>("O")                                                                                         \
      .Attr<FfiInt64Span>("U")                                                                                         \
      .Attr<FfiInt64Span>("qs_offsets")                                                                                \
      .Attr<FfiInt64Span>("A_offsets")                                                                                 \
      .Attr<FfiInt64Span>("B_offsets")                                                                                 \
      .Attr<FfiInt64Span>("C_offsets")                                                                                 \
      .Attr<FfiInt64Span>("I_offsets")                                                                                 \
      .Attr<FfiInt64Span>("I_depths")                                                                                  \
      .Attr<FfiInt64Span>("A_dep_flat")                                                                                \
      .Attr<FfiInt64Span>("A_dep_offsets")                                                                             \
      .Attr<FfiInt64Span>("B_dep_flat")                                                                                \
      .Attr<FfiInt64Span>("B_dep_offsets")                                                                             \
      .Attr<int32_t>("flags")

extern "C" XLA_FFI_Error* pymdp_neg_efe_all_policies(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind() PYMDP_NEG_EFE_ARGS_AND_ATTRS().To(::pymdp_ffi::NegEfeCpu).release();
  return handler->Call(call_frame);
}

#ifdef PYMDP_FFI_HAS_CUDA
// The two NegEfeState multi-stage targets (host-buffer / device-buffer) share
// the same BindInstantiate body; only the Execute trampolines below differ
// (platform tag, Ctx chain). Keep both *_instantiate symbols so the Python
// registration code in _efe.py can wire each target to its own pair.
#define PYMDP_NEG_EFE_INSTANTIATE_TRAMPOLINE(symbol)                                                                   \
  extern "C" XLA_FFI_Error* symbol(XLA_FFI_CallFrame* call_frame) {                                                    \
    static auto* handler = ::xla::ffi::Ffi::BindInstantiate().To(::pymdp_ffi::NegEfeCudaInstantiate).release();        \
    return handler->Call(call_frame);                                                                                  \
  }

// Host-buffer entry (NegEfeCudaHost). Platform tag is "cpu" — XLA hands us
// host buffers; GPU dispatch is internal. Used wherever JAX runs on CPU.
PYMDP_NEG_EFE_INSTANTIATE_TRAMPOLINE(pymdp_neg_efe_all_policies_cuda_host_instantiate)

extern "C" XLA_FFI_Error* pymdp_neg_efe_all_policies_cuda_host(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::State<::pymdp_ffi::NegEfeState>>() PYMDP_NEG_EFE_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::NegEfeCudaHost)
                             .release();
  return handler->Call(call_frame);
}

// Device-buffer entry (NegEfeCudaDev). Platform="CUDA" — XLA hands us
// device-pointer Buffers + the stream. Bind chain prepends
// Ctx<PlatformStream<cudaStream_t>> + Ctx<State<NegEfeState>>.
PYMDP_NEG_EFE_INSTANTIATE_TRAMPOLINE(pymdp_neg_efe_cuda_dev_instantiate)

extern "C" XLA_FFI_Error* pymdp_neg_efe_cuda_dev_execute(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Ctx<ffi::State<::pymdp_ffi::NegEfeState>>() PYMDP_NEG_EFE_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::NegEfeCudaDev)
                             .release();
  return handler->Call(call_frame);
}

#undef PYMDP_NEG_EFE_INSTANTIATE_TRAMPOLINE

// NegEfeState type registration accessors. Python wraps each address in a
// PyCapsule for jax.ffi.register_ffi_type. MakeTypeInfo's deleter uses
// `delete` on NegEfeState*; NegEfeCudaInstantiate returns unique_ptr so the
// contract holds.
extern "C" XLA_FFI_TypeId* pymdp_neg_efe_state_type_id() {
  return &::pymdp_ffi::NegEfeState::id;
}

extern "C" XLA_FFI_TypeInfo* pymdp_neg_efe_state_type_info() {
  static XLA_FFI_TypeInfo info = ffi::MakeTypeInfo<::pymdp_ffi::NegEfeState>();
  return &info;
}
#endif  // PYMDP_FFI_HAS_CUDA

#undef PYMDP_NEG_EFE_ARGS_AND_ATTRS

// Generalized FPI kernel. Shared bind-chain tail used by both the CPU target
// and the CUDA-host shim below.
#define PYMDP_FPI_ARGS_AND_ATTRS()                                                                                     \
  .Arg<FfiF32Buf>()     /* ll_flat */                                                                                  \
      .Arg<FfiF32Buf>() /* lp_flat */                                                                                  \
      .Ret<FfiF32Buf>() /* q_out */                                                                                    \
      .Attr<FfiInt64Span>("S")                                                                                         \
      .Attr<FfiInt64Span>("ll_offsets")                                                                                \
      .Attr<FfiInt64Span>("lp_offsets")                                                                                \
      .Attr<FfiInt64Span>("A_dep_flat")                                                                                \
      .Attr<FfiInt64Span>("A_dep_offsets")                                                                             \
      .Attr<int32_t>("num_iter")

extern "C" XLA_FFI_Error* pymdp_fpi(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind() PYMDP_FPI_ARGS_AND_ATTRS().To(::pymdp_ffi::FpiCpu).release();
  return handler->Call(call_frame);
}

#ifdef PYMDP_FFI_HAS_CUDA
// platform="CUDA" target: D2H inputs from JAX device buffers, run the CPU
// kernel, H2D output back. Used on hosts where JAX's default backend is
// CUDA so the JAX-scan-on-GPU fallback (16+ per-iter launches) is replaced
// by a single FFI dispatch with two transfers around the CPU body.
extern "C" XLA_FFI_Error* pymdp_fpi_cuda_host(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>() PYMDP_FPI_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::FpiCudaHost)
                             .release();
  return handler->Call(call_frame);
}

// platform="CUDA" target: native CUDA FPI. Single kernel runs all num_iter
// iterations on the GPU stream — no host roundtrip, no shim sync. Picked at
// trace time when every modality is K<=3; otherwise the dispatch in
// pymdp/ffi/_fpi.py routes through pymdp_fpi_cuda_host.
extern "C" XLA_FFI_Error* pymdp_fpi_cuda_native(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>() PYMDP_FPI_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::FpiCudaDevice)
                             .release();
  return handler->Call(call_frame);
}
#endif  // PYMDP_FFI_HAS_CUDA

#undef PYMDP_FPI_ARGS_AND_ATTRS

extern "C" int32_t pymdp_ffi_cpu_capabilities() {
  return ::pymdp_ffi::kCpuCapabilities;
}
