// Tegra zero-copy aliasing for JAX device pointers.
//
// On integrated GPUs (Tegra) CPU and GPU share physical
// DRAM. JAX's CUDA backend on Tegra typically allocates "device" buffers as
// either managed memory (cudaMallocManaged) or pinned host memory
// (cudaHostAlloc) — both are directly host-accessible. When that is the case,
// host-side code can read/write the buffer through a host pointer with no
// D2H/H2D copy required.
//
// On discrete GPUs the buffer lives in dedicated VRAM, no host pointer
// exists, and the caller must fall back to cudaMemcpyAsync.
//
// try_alias_as_host returns:
//   * the device pointer itself when attr.type == cudaMemoryTypeManaged
//     (managed memory: same virtual address is valid on both sides on
//     coherent platforms; on Tegra it's hardware-coherent)
//   * attr.hostPointer when attr.type == cudaMemoryTypeHost (pinned host
//     allocation: a distinct host alias mapped to the same physical pages)
//   * nullptr when attr.type is cudaMemoryTypeDevice or the query fails
//     (discrete VRAM or pre-CUDA-11 unregistered memory; caller copies)
//
// cudaPointerGetAttributes costs a few µs on the steady-state path (it
// looks up the allocation in the driver's tracker) but produces no kernel
// launches and no syncs. For Tegra-resident JAX backends it pays for itself
// many times over by removing the cudaMemcpyAsync per-call overhead.
//
// Lifetime note: callers must keep the original device pointer alive for the
// duration of any host-side reads/writes — XLA owns the buffer and we only
// borrow it. That's already true for the FFI ABI (XLA guarantees inputs/
// outputs stay live for the call).

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/error_helpers.h"

namespace pymdp_ffi {

// One-shot diagnostic gated on PYMDP_FFI_TIME=1. Logs the first observed
// memory type so users can confirm whether the alias path fires under their
// JAX allocator setup. Cheap: one atomic CAS after the first call.
inline void log_alias_path_once(cudaMemoryType type, void* result) {
  if (const char* env = std::getenv("PYMDP_FFI_TIME"); !env || env[0] != '1') return;
  static std::atomic<bool> reported{false};
  bool                     expected = false;
  if (!reported.compare_exchange_strong(expected, true)) return;
  const char* type_str = (type == cudaMemoryTypeManaged)        ? "managed"
                         : (type == cudaMemoryTypeHost)         ? "host (pinned)"
                         : (type == cudaMemoryTypeDevice)       ? "device (no alias)"
                         : (type == cudaMemoryTypeUnregistered) ? "unregistered (no alias)"
                                                                : "unknown";
  std::fprintf(stderr, "[ffi-alias] first call: memory type = %s, alias %s\n", type_str,
               result != nullptr ? "HIT (zero-copy)" : "MISS (D2H/H2D fallback)");
}

// Process-level negative cache. JAX's CUDA backend uses one allocator
// strategy per process — once we observe a device-only allocation, every
// subsequent buffer will be the same. Collapses the steady-state alias probe
// from a cudaPointerGetAttributes syscall to a single relaxed atomic load
// when aliasing isn't supported. Note this fires even on some integrated
// (Tegra) backends, where JAX's allocator can default to device-only
// cudaMalloc rather than managed/pinned memory.
inline std::atomic<bool>& alias_known_unsupported() {
  static std::atomic<bool> flag{false};
  return flag;
}

inline void* try_alias_as_host(const void* dev_ptr) {
  if (dev_ptr == nullptr) return nullptr;
  if (alias_known_unsupported().load(std::memory_order_relaxed)) return nullptr;
  cudaPointerAttributes attr{};
  cudaError_t           rc = cudaPointerGetAttributes(&attr, dev_ptr);
  if (rc != cudaSuccess) {
    cudaGetLastError();  // clear the sticky error so it doesn't bleed into the next CUDA call
    log_alias_path_once(cudaMemoryTypeUnregistered, nullptr);
    alias_known_unsupported().store(true, std::memory_order_relaxed);
    return nullptr;
  }
  void* result = nullptr;
  if (attr.type == cudaMemoryTypeManaged) {
    // Managed memory: the device pointer is also a valid host pointer.
    // const_cast is sound here — the underlying memory is read/write; the
    // const came from the caller's view, not from the allocation.
    result = const_cast<void*>(dev_ptr);
  } else if (attr.type == cudaMemoryTypeHost) {
    // Pinned host allocation: attr.hostPointer is the host-accessible alias.
    // (May equal dev_ptr on unified-memory systems but not guaranteed; use
    // the explicit field.)
    result = attr.hostPointer;
  }
  log_alias_path_once(attr.type, result);
  if (result == nullptr) {
    // Device-only or unregistered: latch the negative cache so subsequent
    // calls skip the syscall entirely.
    alias_known_unsupported().store(true, std::memory_order_relaxed);
  }
  return result;
}

// Per-buffer alias-or-D2H input helper for CUDA-host shims.
//
// If `src_dev` is host-accessible (managed or pinned), returns the host
// alias via `*out_host` and skips the D2H entirely. Otherwise queues an
// async `cudaMemcpyAsync` D2H into `*dst` (sized up via std::vector::resize)
// and returns `dst->data()`. `n == 0` returns nullptr without touching CUDA.
//
// Caller must sync the stream before reading `*out_host`: even when aliasing
// fires, prior ops queued on the stream may still be writing the buffer.
//
// `kernel_name` is used as the FfiError category on cudaMemcpyAsync failure.
template <typename T>
inline FfiError staged_d2h_or_alias(const char* kernel_name, std::vector<T>* dst, const T* src_dev, std::size_t n,
                                    cudaStream_t stream, const T** out_host) {
  if (n == 0) {
    *out_host = nullptr;
    return FfiError::Success();
  }
  if (void* alias = try_alias_as_host(src_dev)) {
    *out_host = static_cast<const T*>(alias);
    return FfiError::Success();
  }
  dst->resize(n);
  cudaError_t rc = cudaMemcpyAsync(dst->data(), src_dev, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (rc != cudaSuccess) {
    return invalid_arg(kernel_name, std::string("cudaMemcpyAsync D2H failed: ") + cudaGetErrorString(rc));
  }
  *out_host = dst->data();
  return FfiError::Success();
}

// Symmetric per-buffer output helper. If `dst_dev` is host-accessible
// (managed or pinned), the caller has already written through the alias and
// no copy is needed — pass `aliased=true` and this is a no-op. Otherwise
// queues an async `cudaMemcpyAsync` H2D from `src_host` into `dst_dev`.
// `n == 0` short-circuits.
//
// Pair with `staged_d2h_or_alias` on input buffers: a kernel that has
// aliased every buffer can return without any cudaMemcpyAsync, with one
// cudaStreamSynchronize before the kernel covering all aliased reads.
// Failed aliasing on any single buffer triggers the H2D fallback for that
// buffer only — partial aliasing is fine.
//
// `kernel_name` is used as the FfiError category on cudaMemcpyAsync failure.
template <typename T>
inline FfiError staged_h2d(const char* kernel_name, T* dst_dev, const T* src_host, std::size_t n, cudaStream_t stream,
                           bool aliased) {
  if (n == 0 || aliased) return FfiError::Success();
  cudaError_t rc = cudaMemcpyAsync(dst_dev, src_host, n * sizeof(T), cudaMemcpyHostToDevice, stream);
  if (rc != cudaSuccess) {
    return invalid_arg(kernel_name, std::string("cudaMemcpyAsync H2D failed: ") + cudaGetErrorString(rc));
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
