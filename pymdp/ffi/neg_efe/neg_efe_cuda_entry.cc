// ABI entries for the CUDA neg-EFE kernel.
//
// Two XLA registrations share the cache/runtime pipeline (cache.cc + runtime.cc
// + launch.cc); selection is made from Python via PYMDP_FFI_USE_CUDA + presence
// of a CUDA JAX backend:
//   * NegEfeCudaHost (platform="cpu")  — host buffers in/out, used wherever
//     JAX runs on CPU. D2H copy at the end.
//   * NegEfeCudaDev  (platform="CUDA") — JAX device buffers in/out, scatter
//     writes straight into JAX's output, no D2H.
//
// All device-visible storage is cudaMallocManaged; on Tegra that maps to
// shared DRAM so host packing + a single device sync covers the
// single-stream pipeline.

#include "neg_efe/neg_efe.h"

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/cuda_host_alias.h"
#include "common/cuda_memory.h"
#include "common/error_helpers.h"
#include "neg_efe/neg_efe_cuda_context.h"
#include "neg_efe/neg_efe_cuda_internal.h"
#include "neg_efe/neg_efe_entry.h"
#include "neg_efe/neg_efe_layout.h"

namespace ffi = ::xla::ffi;

namespace pymdp_ffi {
namespace {

// True when all six caches still match the current devptrs / sigs.
bool caches_are_warm(const NegEfeContext& ctx, const Layout& L, int Bn, KernelFlags flags, bool pA_present,
                     bool pB_present, const DevSrcs& devs) {
  const uint64_t pm_sig       = factor_history_pm_sig(L);
  const int64_t  pm_size      = static_cast<int64_t>(Bn) * L.P * L.T * L.F;
  const uint64_t a_sig        = a_sig_bn(L, Bn);
  const int64_t  a_size       = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t b_sig        = b_sig_bn(L, Bn);
  const int64_t  b_size       = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t linear_sig   = cuda_linear_sig(L, Bn);
  const int32_t  linear_flags = (flags.use_states_info_gain ? 1 : 0) | (flags.use_utility ? 2 : 0);

  if (!ctx.tree_cache.match_devptr(devs.pm, pm_size, pm_sig)) return false;
  if (!ctx.a_cache.match_devptr(devs.A, a_size, a_sig)) return false;
  if (!ctx.b_cache.match_devptr(devs.B, b_size, b_sig)) return false;
  if (!ctx.linear_cache.match_devptr(devs.A, devs.C, linear_sig, linear_flags)) return false;
  if (flags.use_param_info_gain && pA_present && !ctx.wa_cache.match_devptr(devs.pA, a_size, a_sig)) return false;
  if (flags.use_param_info_gain && pB_present && !ctx.wb_cache.match_devptr(devs.pB, b_size, b_sig)) return false;
  return true;
}

}  // namespace

// Host-buffer ABI (platform="cpu"). GPU work runs through managed scratch;
// scatter writes ctx.scratch.out_dev (owned) and D2H copies into the FFI
// out buffer at the end.

FfiError NegEfeCudaHost(NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B,
                        FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                        FfiF32Out out, FfiInt64Span S_span, FfiInt64Span O_span, FfiInt64Span U_span,
                        FfiInt64Span qs_off_span, FfiInt64Span A_off_span, FfiInt64Span B_off_span,
                        FfiInt64Span C_off_span, FfiInt64Span I_off_span, FfiInt64Span I_depths_span,
                        FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span, FfiInt64Span B_dep_flat_span,
                        FfiInt64Span B_dep_off_span, int32_t flags) {
  const LayoutSpans spans{S_span,         O_span,          U_span,        qs_off_span,   A_off_span,
                          B_off_span,     C_off_span,      I_off_span,    I_depths_span, A_dep_flat_span,
                          A_dep_off_span, B_dep_flat_span, B_dep_off_span};
  ParsedCall        pc;
  PYMDP_TRY(
      parse_and_validate_call(policy_matrix, qs_init, A, B, C, I, pA, pB, inductive_epsilon, out, spans, flags, &pc));

  NegEfeContext& ctx    = *state->ctx;
  const int      Bn_int = static_cast<int>(pc.shape.Bn);

  PYMDP_TRY(prepare_caches(ctx, pc.L, pc.flags, Bn_int, policy_matrix.typed_data(), A.typed_data(), B.typed_data(),
                           C.typed_data(), pA.typed_data(), pB.typed_data(), pc.pA_present, pc.pB_present));

  log_efe_cuda_timestats_if_enabled(ctx, pc.L, pc.shape.Bn);

  PYMDP_TRY(ensure_forward_scratch(ctx, pc.L, pc.flags, Bn_int, /*stream=*/nullptr, /*out_is_external=*/false));
  upload_qs_init(ctx, pc.L, Bn_int, qs_init.typed_data());
  if (pc.flags.use_inductive) {
    PYMDP_TRY(upload_inductive_vector(ctx, pc.L, Bn_int, qs_init.typed_data(), I.typed_data(),
                                      inductive_epsilon.typed_data(), pc.epsilon));
  }
  return run_forward(ctx, pc.L, pc.flags, Bn_int, ctx.scratch.out_dev.as<float>(), out->typed_data());
}

// Device-buffer ABI (platform="CUDA"). JAX passes device buffers + stream;
// scatter writes JAX's output directly. Warm path (all caches match the
// prior call's devptrs) skips D2H entirely; cold path does D2H + content-tag
// cache fill, then records the new devptrs.

ffi::TypeId NegEfeState::id = {};

NegEfeState::NegEfeState() : ctx(std::make_unique<NegEfeContext>()) {}
NegEfeState::~NegEfeState() = default;

ffi::ErrorOr<std::unique_ptr<NegEfeState>> NegEfeCudaInstantiate() {
  return std::make_unique<NegEfeState>();
}

FfiError NegEfeCudaDev(cudaStream_t stream, NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A,
                       FfiF32Buf B, FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                       FfiF32Out out, FfiInt64Span S_span, FfiInt64Span O_span, FfiInt64Span U_span,
                       FfiInt64Span qs_off_span, FfiInt64Span A_off_span, FfiInt64Span B_off_span,
                       FfiInt64Span C_off_span, FfiInt64Span I_off_span, FfiInt64Span I_depths_span,
                       FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span, FfiInt64Span B_dep_flat_span,
                       FfiInt64Span B_dep_off_span, int32_t flags) {
  const LayoutSpans spans{S_span,         O_span,          U_span,        qs_off_span,   A_off_span,
                          B_off_span,     C_off_span,      I_off_span,    I_depths_span, A_dep_flat_span,
                          A_dep_off_span, B_dep_flat_span, B_dep_off_span};
  ParsedCall        pc;
  PYMDP_TRY(
      parse_and_validate_call(policy_matrix, qs_init, A, B, C, I, pA, pB, inductive_epsilon, out, spans, flags, &pc));

  NegEfeContext& ctx    = *state->ctx;
  const int      Bn_int = static_cast<int>(pc.shape.Bn);

  const DevSrcs devs{
      .pm = policy_matrix.typed_data(),
      .A  = A.typed_data(),
      .B  = B.typed_data(),
      .C  = pc.flags.use_utility ? static_cast<const void*>(C.typed_data()) : nullptr,
      .pA = pc.pA_present ? static_cast<const void*>(pA.typed_data()) : nullptr,
      .pB = pc.pB_present ? static_cast<const void*>(pB.typed_data()) : nullptr,
  };

  // When param_info_gain is on but pA / pB is absent, the warm-check skips
  // the absent slot — clear any stale prior-call entry so downstream
  // `!empty()` guards read absence correctly.
  if (pc.flags.use_param_info_gain) {
    if (!pc.pA_present) ctx.wa_cache.clear();
    if (!pc.pB_present) ctx.wb_cache.clear();
  }

  if (!caches_are_warm(ctx, pc.L, Bn_int, pc.flags, pc.pA_present, pc.pB_present, devs)) {
    // Tegra zero-copy on the cold-cache path. For each input buffer try the
    // host alias first (managed/pinned memory has a host address); only D2H
    // when JAX hands us device-only memory. The stream sync below is still
    // required either way — prior queued ops may still be writing these
    // buffers.
    thread_local std::vector<int32_t> pm_host;
    thread_local std::vector<float>   A_host, B_host, C_host, pA_host, pB_host;
    const int32_t*                    pm_ptr = nullptr;
    const float*                      A_ptr  = nullptr;
    const float*                      B_ptr  = nullptr;
    const float*                      C_ptr  = nullptr;
    const float*                      pA_ptr = nullptr;
    const float*                      pB_ptr = nullptr;
    PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &pm_host, policy_matrix.typed_data(),
                                  policy_matrix.element_count(), stream, &pm_ptr));
    PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &A_host, A.typed_data(), A.element_count(), stream, &A_ptr));
    PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &B_host, B.typed_data(), B.element_count(), stream, &B_ptr));
    if (pc.flags.use_utility)
      PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &C_host, C.typed_data(), C.element_count(), stream, &C_ptr));
    if (pc.pA_present)
      PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &pA_host, pA.typed_data(), pA.element_count(), stream, &pA_ptr));
    if (pc.pB_present)
      PYMDP_TRY(staged_d2h_or_alias(kEfeKernelName, &pB_host, pB.typed_data(), pB.element_count(), stream, &pB_ptr));
    if (cudaError_t rc = cudaStreamSynchronize(stream); rc != cudaSuccess) {
      return invalid_arg(kEfeKernelName,
                         std::string("cudaStreamSynchronize before host work failed: ") + cudaGetErrorString(rc));
    }
    PYMDP_TRY(prepare_caches(ctx, pc.L, pc.flags, Bn_int, pm_ptr, A_ptr, B_ptr, C_ptr, pA_ptr, pB_ptr, pc.pA_present,
                             pc.pB_present, devs));
  }

  // JAX's output device buffer flows straight through StageCtx.out_dev_ptr
  // into launch_final_scatter — never stored in a CuArr, so ensure() / reset()
  // can't free a JAX-owned pointer.
  if (static_cast<int64_t>(out->element_count()) != pc.shape.Bn * pc.shape.P) {
    return invalid_arg(kEfeKernelName, "out element_count " + std::to_string(out->element_count()) +
                                           " != Bn*P = " + std::to_string(pc.shape.Bn * pc.shape.P));
  }

  PYMDP_TRY(ensure_forward_scratch(ctx, pc.L, pc.flags, Bn_int, stream, /*out_is_external=*/true));
  PYMDP_TRY(upload_qs_init_d2d(ctx, pc.L, Bn_int, qs_init.typed_data(), stream));
  if (pc.flags.use_inductive) {
    PYMDP_TRY(upload_inductive_vector_d2d(ctx, pc.L, Bn_int, qs_init.typed_data(), I.typed_data(),
                                          inductive_epsilon.typed_data(), pc.epsilon, stream));
  }
  PYMDP_TRY(run_forward(ctx, pc.L, pc.flags, Bn_int, out->typed_data(), /*out_host_or_null=*/nullptr, stream));
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
