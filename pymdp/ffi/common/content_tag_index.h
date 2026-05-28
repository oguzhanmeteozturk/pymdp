// Content-tag sampling layout: the dimension constants and the per-slot
// stride-sample index. Split out from cache_lru.h so the CUDA gather kernel
// (neg_efe_cuda_kernels.cu) can share one definition with the host hash
// without dragging in cache_lru.h's C++17 `inline constexpr` variables — the
// device TU compiles at C++14 (nvcc 10.2 on jetson-nano caps device dialect).
//
// Keep this header C++14-clean and free of host-only / XLA-FFI includes: it is
// compiled by nvcc for device code.

#pragma once

#include <cstdint>

namespace pymdp_ffi {

// Sampling layout: a 16-element prefix plus 8 strided positions, concatenated
// into a single 24-entry buffer (prefix at [0, kContentTagPrefixMax), stride
// samples at [kContentTagPrefixMax, kContentTagTotalSamples)). Plain
// `constexpr` (not `inline constexpr`) keeps this valid under C++14.
constexpr int kContentTagStrideSamples = 8;
constexpr int kContentTagPrefixMax     = 16;
constexpr int kContentTagTotalSamples  = kContentTagPrefixMax + kContentTagStrideSamples;

// Index of the slot-th stride sample within a buffer of length `size`.
// Marked host+device so the content-tag gather kernel and the host hash
// sample identical positions from one definition. (When not compiled by nvcc
// the qualifier expands to nothing.)
#ifdef __CUDACC__
#define PYMDP_CONTENT_TAG_HD __host__ __device__
#else
#define PYMDP_CONTENT_TAG_HD
#endif
PYMDP_CONTENT_TAG_HD inline int64_t content_tag_stride_index(int slot, int64_t size) {
  int64_t idx = 0;
  switch (slot) {
    case 0: idx = 0; break;
    case 1: idx = (size > 1) ? 1 : 0; break;
    case 2: idx = size / 8; break;
    case 3: idx = size / 4; break;
    case 4: idx = size / 2; break;
    case 5: idx = (size * 3) / 4; break;
    case 6: idx = (size * 7) / 8; break;
    case 7: idx = size - 1; break;
    default: idx = 0; break;
  }
  if (idx < 0) idx = 0;
  if (idx >= size) idx = size - 1;
  return idx;
}
#undef PYMDP_CONTENT_TAG_HD

}  // namespace pymdp_ffi
