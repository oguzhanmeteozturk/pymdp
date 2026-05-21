// Shared CUDA device-memory primitives + error helpers.
//
// CuArr: owned (or non-owning view) device allocation backed by
// cudaMallocManaged. Owns = false views alias into a CuPool slab.
// NOTE: cudaMallocManaged on pre-Pascal (Maxwell sm_53) does not support
// concurrent CPU+GPU access — never touch host-side managed memory while
// a kernel is in flight.
//
// CuPool: bump-allocator over a single CuArr slab. Used for model-parameter
// caches so one cudaMalloc covers all per-modality/factor arrays.
//
// `cuda_err` / `cublas_err` take an explicit kernel-name category so the
// FfiError ends up tagged with the calling kernel ("efe_ffi" / "fpi_ffi"),
// not whichever one happens to be in scope. The CUDA_TRY / CUBLAS_TRY
// macros aren't defined here — each TU plugs in its own one-line define
// that fixes the kernel name (see the pattern in any CUDA-using .cc).

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstddef>
#include <cstring>
#include <string>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "common/error_helpers.h"

namespace pymdp_ffi {

inline FfiError cuda_err(const char* kernel_name, const char* op, cudaError_t rc) {
  if (rc == cudaSuccess) return FfiError::Success();
  return invalid_arg(kernel_name, std::string("cuda ") + op + " failed: " + cudaGetErrorString(rc));
}

inline FfiError cublas_err(const char* kernel_name, const char* op, cublasStatus_t st) {
  if (st == CUBLAS_STATUS_SUCCESS) return FfiError::Success();
  return invalid_arg(kernel_name, std::string("cublas ") + op + " failed");
}

inline size_t round_up_8(size_t n) {
  return (n + 7) & ~static_cast<size_t>(7);
}

// True if the CUDA runtime is currently unloading. Destructor cleanup order
// at process exit isn't fixed — static / thread_local / FFI-state-owned
// destructors can fire after the CUDA runtime has begun shutdown, at which
// point cudaFree / cublasDestroy on managed pointers can SEGV on Tegra
// (managed-memory teardown + early-driver-unload race on Tegra at process
// exit). When this returns true, skip the cleanup — the OS reclaims
// everything when the process dies. Probing the runtime via cudaGetDevice
// is the canonical signal: it returns cudaErrorCudartUnloading once the
// unload flag is set.
inline bool cuda_runtime_unloading() noexcept {
  int               dev = -1;
  const cudaError_t rc  = cudaGetDevice(&dev);
  return rc == cudaErrorCudartUnloading;
}

struct CuArr {
  void*  ptr   = nullptr;
  size_t bytes = 0;
  bool   owns  = true;

  CuArr() = default;
  CuArr(void* p, size_t b) : ptr(p), bytes(b), owns(true) {}
  static CuArr view(void* p, size_t b) {
    CuArr a;
    a.ptr   = p;
    a.bytes = b;
    a.owns  = false;
    return a;
  }
  CuArr(CuArr&& o) noexcept : ptr(o.ptr), bytes(o.bytes), owns(o.owns) {
    o.ptr   = nullptr;
    o.bytes = 0;
    o.owns  = true;
  }
  CuArr& operator=(CuArr&& o) noexcept {
    if (this != &o) {
      reset();
      ptr     = o.ptr;
      bytes   = o.bytes;
      owns    = o.owns;
      o.ptr   = nullptr;
      o.bytes = 0;
      o.owns  = true;
    }
    return *this;
  }
  CuArr(const CuArr&)            = delete;
  CuArr& operator=(const CuArr&) = delete;
  ~CuArr() { reset(); }

  void reset() {
    if (ptr && owns && !cuda_runtime_unloading()) {
      cudaFree(ptr);
    }
    ptr   = nullptr;
    bytes = 0;
    owns  = true;
  }

  cudaError_t ensure(size_t n_bytes) {
    if (bytes >= n_bytes) return cudaSuccess;
    reset();
    if (n_bytes == 0) return cudaSuccess;
    void*       p  = nullptr;
    cudaError_t rc = cudaMallocManaged(&p, n_bytes);
    if (rc != cudaSuccess) return rc;
    ptr   = p;
    bytes = n_bytes;
    owns  = true;
    return cudaSuccess;
  }

  template <class T> T* as() const { return static_cast<T*>(ptr); }
};

struct CuPool {
  CuArr  backing;
  size_t used = 0;

  void reset() {
    backing.reset();
    used = 0;
  }

  cudaError_t reserve(size_t need_bytes) {
    used = 0;
    return backing.ensure(need_bytes);
  }

  CuArr append_copy(const void* src, size_t n_bytes) {
    constexpr size_t kAlign = 8;
    used                    = (used + (kAlign - 1)) & ~(kAlign - 1);
    if (n_bytes == 0) return CuArr{};
    char* dst = static_cast<char*>(backing.ptr) + used;
    std::memcpy(dst, src, n_bytes);
    CuArr view = CuArr::view(dst, n_bytes);
    used += n_bytes;
    return view;
  }
};

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
