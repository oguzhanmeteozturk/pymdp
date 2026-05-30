// FpiCudaDevice — native CUDA FPI ABI entry (platform="CUDA").
//
// Single CUDA kernel runs all `num_iter` iterations internally, one block
// per batch element. No D2H, no H2D, no host-side stream sync — JAX's
// surrounding ops on the same stream pipeline naturally with the FPI
// kernel.
//
// Restrictions: every modality's A_dependencies rank must be in [1, 3].
// The host-side gate in pymdp/ffi/_fpi.py enforces this; calls landing
// here with K>=4 hit the kernel's switch default and produce silently
// wrong output (the gate is the contract). K>=4 dispatches go through
// FpiCudaHost (shim) or the JAX scan reference instead.
//
// Per-call host work: refresh the dispatch-table cache (fpi_cuda_cache),
// runtime shape gates (batch derivation + buffer count checks), then
// launch the chosen kernel variant.

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <cstdlib>
#include <string>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "fpi/fpi.h"
#include "fpi/fpi_cuda_cache.h"
#include "fpi/fpi_cuda_context.h"
#include "fpi/fpi_cuda_kernels.h"
#include "common/cuda_memory.h"
#include "fpi/fpi_entry.h"

#define CUDA_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cuda_err(::pymdp_ffi::kFpiKernelName, op, (expr)))

namespace pymdp_ffi {

FfiError FpiCudaDevice(cudaStream_t stream, FfiF32Buf ll_flat_dev, FfiF32Buf lp_flat_dev, FfiF32Out q_out_dev,
                       FfiInt64Span S_span, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                       FfiInt64Span A_dep_offsets, int32_t num_iter) {
  const int64_t F = static_cast<int64_t>(S_span.size());
  const int64_t M = static_cast<int64_t>(A_dep_offsets.size()) - 1;

  // Cheap span-bounds guard so the cache helper can read the spans
  // without UB. Full attr validation (monotonicity, positivity, K range)
  // is gated by the cache miss path; cache hits skip it.
  if (F <= 0 || M <= 0) {
    return invalid_arg(kFpiKernelName, "invalid F=" + std::to_string(F) + " or M=" + std::to_string(M));
  }
  if (static_cast<int64_t>(lp_offsets.size()) != F + 1 || static_cast<int64_t>(ll_offsets.size()) != M + 1) {
    return invalid_arg(kFpiKernelName, "lp_offsets/ll_offsets span size mismatch with F/M");
  }

  FpiCudaDeviceScratch* cs            = nullptr;
  bool                  use_smallmeta = false;
  PYMDP_TRY(refresh_fpi_cuda_cache(S_span, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets, num_iter, F, M, stream,
                                   &cs, &use_smallmeta));

  // Runtime-shape checks: always run. lp_flat_dev / ll_flat_dev /
  // q_out_dev element counts come from the JAX runtime (batch dim under
  // vmap) and are not part of the static attrs — same model metadata may
  // be called with different batch sizes.
  const int64_t total_S  = lp_offsets[F];
  const int64_t total_ll = ll_offsets[M];
  const int64_t lp_count = lp_flat_dev.element_count();
  int64_t       batch    = 0;
  PYMDP_TRY(validate_fpi_batch_shapes(lp_count, ll_flat_dev.element_count(), q_out_dev->element_count(), total_S,
                                      total_ll, &batch));

  // Diagnostic bypass (PYMDP_FFI_FPI_KERNEL_NOOP=1): launch empty kernel
  // with matching grid/block/shmem. Bench delta vs real launch isolates
  // kernel-internal work from XLA-dispatch + driver overhead.
  static const bool kFpiNoopMode = []() {
    const char* v = std::getenv("PYMDP_FFI_FPI_KERNEL_NOOP");
    return v != nullptr && v[0] != '\0' && v[0] != '0';
  }();
  if (kFpiNoopMode) {
    CUDA_TRY("fpi_cuda noop launch",
             fpi_cuda::launch_fpi_noop(static_cast<int>(batch), static_cast<int>(total_S), stream));
    return FfiError::Success();
  }

  if (use_smallmeta) {
    CUDA_TRY("fpi_cuda launch (smallmeta)",
             fpi_cuda::launch_fpi_smallmeta(ll_flat_dev.typed_data(), lp_flat_dev.typed_data(), q_out_dev->typed_data(),
                                            static_cast<int>(batch), static_cast<int>(F), static_cast<int>(M),
                                            static_cast<int>(total_ll), static_cast<int>(total_S), num_iter,
                                            cs->smallmeta, cs->sync_mask, stream));
  } else {
    CUDA_TRY("fpi_cuda launch",
             fpi_cuda::launch_fpi(ll_flat_dev.typed_data(), lp_flat_dev.typed_data(), q_out_dev->typed_data(),
                                  static_cast<int>(batch), static_cast<int>(F), static_cast<int>(M),
                                  static_cast<int>(total_ll), static_cast<int>(total_S), num_iter,
                                  cs->S_dev.as<const int32_t>(), cs->lp_offsets_dev.as<const int32_t>(),
                                  cs->mods_dev.as<const fpi_cuda::ModalityDispatchGpu>(), cs->sync_mask, stream));
  }

  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
