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

#include "neg_efe.h"
#include "neg_efe_layout.h"
#include "fpi.h"
#include "kernel_primitives.h"

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
// Host-buffer CUDA entry (NegEfeCudaHost). Platform tag is "cpu" — XLA
// hands us host buffers; GPU dispatch is internal to the kernel. Used on
// Jetson Nano (CPU-only jaxlib) and x86 with `JAX_PLATFORMS=cpu`. Multi-stage
// so the per-JIT NegEfeState owns the model-parameter caches.
extern "C" XLA_FFI_Error* pymdp_neg_efe_all_policies_cuda_host_instantiate(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::BindInstantiate().To(::pymdp_ffi::NegEfeCudaInstantiate).release();
  return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* pymdp_neg_efe_all_policies_cuda_host(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::State<::pymdp_ffi::NegEfeState>>() PYMDP_NEG_EFE_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::NegEfeCudaHost)
                             .release();
  return handler->Call(call_frame);
}

// =============================================================================
// Device-buffer CUDA entry (NegEfeCudaDev). Platform="CUDA" — XLA hands us
// device-pointer Buffers + the stream. Multi-stage: BindInstantiate creates
// a per-jit-instance NegEfeState (lifetime = compiled XLA executable);
// Execute is the hot path. Bind chain prepends Ctx<PlatformStream<cudaStream_t>>
// + Ctx<State<NegEfeState>> ahead of the shared ABI.
//
// NegEfeState type is registered on the Python side via
// jax.ffi.register_ffi_type; pymdp/ffi/_core.py:_register_multistage wires
// up the type-id and type-info pointers exposed below.
// =============================================================================

extern "C" XLA_FFI_Error* pymdp_neg_efe_cuda_dev_instantiate(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::BindInstantiate().To(::pymdp_ffi::NegEfeCudaInstantiate).release();
  return handler->Call(call_frame);
}

extern "C" XLA_FFI_Error* pymdp_neg_efe_cuda_dev_execute(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Ctx<ffi::PlatformStream<cudaStream_t>>()
                             .Ctx<ffi::State<::pymdp_ffi::NegEfeState>>() PYMDP_NEG_EFE_ARGS_AND_ATTRS()
                             .To(::pymdp_ffi::NegEfeCudaDev)
                             .release();
  return handler->Call(call_frame);
}

// Accessors for NegEfeState type registration. Python reads these once and
// wraps each address in a PyCapsule for jax.ffi.register_ffi_type.
// MakeTypeInfo's deleter uses delete on NegEfeState*; the unique_ptr from
// NegEfeCudaInstantiate satisfies that contract.
extern "C" XLA_FFI_TypeId* pymdp_neg_efe_state_type_id() {
  return &::pymdp_ffi::NegEfeState::id;
}

extern "C" XLA_FFI_TypeInfo* pymdp_neg_efe_state_type_info() {
  static XLA_FFI_TypeInfo info = ffi::MakeTypeInfo<::pymdp_ffi::NegEfeState>();
  return &info;
}
#endif  // PYMDP_FFI_HAS_CUDA

#undef PYMDP_NEG_EFE_ARGS_AND_ATTRS

// Generalized FPI kernel. Single-target, so no macro layer.
extern "C" XLA_FFI_Error* pymdp_fpi(XLA_FFI_CallFrame* call_frame) {
  static auto* handler = ::xla::ffi::Ffi::Bind()
                             .Arg<FfiF32Buf>()  // ll_flat
                             .Arg<FfiF32Buf>()  // lp_flat
                             .Ret<FfiF32Buf>()  // q_out (single flat buffer)
                             .Attr<FfiInt64Span>("S")
                             .Attr<FfiInt64Span>("ll_offsets")
                             .Attr<FfiInt64Span>("lp_offsets")
                             .Attr<FfiInt64Span>("A_dep_flat")
                             .Attr<FfiInt64Span>("A_dep_offsets")
                             .Attr<int32_t>("num_iter")
                             .To(::pymdp_ffi::FpiCpu)
                             .release();
  return handler->Call(call_frame);
}

extern "C" int32_t pymdp_ffi_cpu_capabilities() {
  return ::pymdp_ffi::kCpuCapabilities;
}
