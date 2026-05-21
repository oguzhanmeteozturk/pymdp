// FpiCudaHost — platform="CUDA" shim that routes the FPI through the
// validated CPU kernel on host-side data.
//
// JAX hands us device-pointer Buffers + the stream. On Tegra (integrated
// GPU, unified memory) JAX typically allocates device buffers as managed or
// pinned memory, both host-accessible, and we read/write through host
// aliases with no D2H/H2D. On dGPU hosts we fall back to a per-buffer
// staged D2H of inputs + staged H2D of outputs.
//
// Uses per-buffer `staged_d2h_or_alias` + `staged_h2d` from
// common/cuda_host_alias.h, allowing partial aliasing (e.g. ll aliases, lp
// does not) instead of an all-or-nothing approach.

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/cuda_host_alias.h"
#include "common/error_helpers.h"
#include "fpi/fpi.h"
#include "fpi/fpi_cpu_runner.h"
#include "fpi/fpi_entry.h"

namespace pymdp_ffi {

FfiError FpiCudaHost(cudaStream_t stream, FfiF32Buf ll_flat_dev, FfiF32Buf lp_flat_dev, FfiF32Out q_out_dev,
                     FfiInt64Span S, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat,
                     FfiInt64Span A_dep_offsets, int32_t num_iter) {
  const int64_t ll_count = ll_flat_dev.element_count();
  const int64_t lp_count = lp_flat_dev.element_count();
  const int64_t q_count  = q_out_dev->element_count();

  // Per-buffer alias-or-D2H for the two inputs. Aliases when JAX allocates
  // managed / pinned memory (Tegra); otherwise stages a D2H into thread_local
  // scratch.
  thread_local std::vector<float> ll_host;
  thread_local std::vector<float> lp_host;
  const float*                    ll_ptr = nullptr;
  const float*                    lp_ptr = nullptr;
  PYMDP_TRY(staged_d2h_or_alias<float>(kFpiKernelName, &ll_host, ll_flat_dev.typed_data(),
                                       static_cast<std::size_t>(std::max<int64_t>(ll_count, 0)), stream, &ll_ptr));
  PYMDP_TRY(staged_d2h_or_alias<float>(kFpiKernelName, &lp_host, lp_flat_dev.typed_data(),
                                       static_cast<std::size_t>(std::max<int64_t>(lp_count, 0)), stream, &lp_ptr));

  // Output: alias when the device buffer is host-accessible; otherwise the
  // kernel writes into thread_local q_host and we H2D below.
  float* q_alias = q_count > 0 ? static_cast<float*>(try_alias_as_host(q_out_dev->typed_data())) : nullptr;
  thread_local std::vector<float> q_host;
  if (q_alias == nullptr && q_count > 0) {
    q_host.resize(static_cast<std::size_t>(q_count));
  }
  float* q_ptr = (q_alias != nullptr) ? q_alias : (q_count > 0 ? q_host.data() : nullptr);

  // One stream sync covers both alias reads (we must wait for prior ops on
  // the stream to finish writing ll/lp before the CPU kernel reads them)
  // and any queued D2H async copies above.
  if (cudaError_t rc = cudaStreamSynchronize(stream); rc != cudaSuccess) {
    return invalid_arg(kFpiKernelName, std::string("cudaStreamSynchronize failed: ") + cudaGetErrorString(rc));
  }

  PYMDP_TRY(run_fpi_kernel_host(ll_ptr, ll_count, lp_ptr, lp_count, q_ptr, q_count, S, ll_offsets, lp_offsets,
                                A_dep_flat, A_dep_offsets, num_iter));

  // Queue the H2D back to JAX's device output only when we didn't alias
  // through it. staged_h2d short-circuits in the aliased / zero-count case.
  // No post-H2D sync needed: stream ordering guarantees any op JAX queues
  // after our return on this stream sees q_out written. Pageable H2D is
  // implicitly synchronous wrt the source host buffer (cudaMemcpyAsync
  // stages from `q_host` before returning), so the thread_local resize on
  // the next call doesn't race the in-flight DMA.
  return staged_h2d<float>(kFpiKernelName, q_out_dev->typed_data(), q_host.data(),
                           static_cast<std::size_t>(std::max<int64_t>(q_count, 0)), stream,
                           /*aliased=*/q_alias != nullptr);
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
