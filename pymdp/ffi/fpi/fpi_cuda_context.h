// Per-thread device-side state for the FPI CUDA kernel.
//
// Holds the dispatch-table device buffers (S, lp_offsets, mods) for the
// pointer-fed launch path, the by-value `FpiSmallMeta` for the cmem path,
// and a two-tier attr-span cache key (pointer-identity + FNV-1a sig).
//
// Reformulating this single-slot cache onto `common/cache_lru.h::CacheLRU`
// is structurally awkward — CacheLRU keys off a (const float*, size,
// layout_sig, content_tag) shape designed for float-buffer payloads with
// sampled content tags, whereas FPI's natural key is "the five int64
// attr-span (ptr, size) tuples plus the layout signature." The
// pointer-identity fast-path here is a real optimization vs. always
// computing the FNV pass (XLA's static-attr buffer is pointer-stable
// across calls of the same compiled executable, so calls 2..N skip the hash
// entirely). Capacity=1 LRU isn't an improvement over the current branchy
// match; widening CacheLRU's key shape to absorb attr spans would touch
// every other caller. Revisit only if a second kernel develops the same
// attr-span cache pattern.

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

  // Pointer-identity fast cache. XLA stores the static FFI attr arrays in
  // a per-executable buffer that is pointer-stable across calls of the
  // same compiled function — so within a rollout the attr spans handed to
  // us alias the same backing memory each call. Comparing pointers +
  // sizes (~6 word loads + 6 compares) skips the FNV-1a pass entirely on
  // a hit. On miss we fall through to the FNV path, which still validates
  // the content cache via `sig`.
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
