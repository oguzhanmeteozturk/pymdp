// Per-thread device-side state for the FPI CUDA kernel.
//
// Holds the dispatch-table device buffers (S, lp_offsets, mods) for the
// pointer-fed launch path, the by-value `FpiSmallMeta` for the cmem path,
// and a two-tier attr-span cache key (pointer-identity + FNV-1a sig).
//
// Not layered on CacheLRU: FPI's natural key is five int64 attr-span
// (ptr, size) tuples + a layout sig, whereas CacheLRU keys off float-buffer
// payloads with sampled content tags — adapting it would touch every other
// caller. The pointer-identity fast-path is a real win because XLA's
// static-attr buffer is pointer-stable across calls of the same compiled
// executable, so calls 2..N skip the FNV pass entirely.

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstddef>
#include <cstdint>

#include "fpi/fpi_cuda_kernels.h"
#include "common/cuda_memory.h"

namespace pymdp_ffi {

// Per-thread scratch holding the device-side dispatch arrays. Sized lazily
// via CuArr::ensure (resize-up-only); after warm-up steady-state cost is
// zero allocation.
struct FpiCudaDeviceScratch {
  // Pointer-fed path (F > kMaxFSmallMeta or M > kMaxMSmallMeta): device
  // buffers populated via H2D on cache miss, kernel reads through global
  // pointers (LDG, L1-cached).
  CuArr S_dev;           // int32_t[F]
  CuArr lp_offsets_dev;  // int32_t[F]
  CuArr mods_dev;        // fpi_cuda::ModalityDispatchGpu[M]
  // Smallmeta path (F <= kMaxFSmallMeta && M <= kMaxMSmallMeta): cached
  // by-value struct passed as a kernel argument, served from the cmem
  // parameter bank (LDC, broadcast). Populated on the same cache-miss
  // event as the device buffers; the active path at launch time is chosen
  // by the F/M dimensions, not by which storage was filled.
  fpi_cuda::FpiSmallMeta smallmeta{};

  // Bitmask over modalities: bit m == 1 iff modality m's A_dependencies
  // overlap modality m+1's, requiring a __syncthreads between them in the
  // kernel. Bit (M-1) is always 0; an unconditional barrier after the
  // modality loop covers the final write. Derived from A_dep_flat /
  // A_dep_offsets on the same cache-miss event as the dispatch table.
  uint32_t sync_mask = 0;

  // Content fingerprint of the *raw attr spans* used to populate the
  // device buffers — S, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets,
  // plus an F/M shape tag. `0` is reserved for "never uploaded yet" so
  // a fresh-process first call always misses.
  uint64_t sig = 0;

  // Pointer-identity fast cache. XLA's static-attr buffer is pointer-stable
  // across calls of the same compiled function, so pointer+size comparison
  // (~6 word loads + 6 compares) skips the FNV-1a pass on a hit. On miss,
  // fall through to the FNV path, which validates the content cache via `sig`.
  const int64_t* last_S_ptr              = nullptr;
  const int64_t* last_ll_offsets_ptr     = nullptr;
  const int64_t* last_lp_offsets_ptr     = nullptr;
  const int64_t* last_A_dep_flat_ptr     = nullptr;
  const int64_t* last_A_dep_offsets_ptr  = nullptr;
  std::size_t    last_S_size             = 0;
  std::size_t    last_ll_offsets_size    = 0;
  std::size_t    last_lp_offsets_size    = 0;
  std::size_t    last_A_dep_flat_size    = 0;
  std::size_t    last_A_dep_offsets_size = 0;
};

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
