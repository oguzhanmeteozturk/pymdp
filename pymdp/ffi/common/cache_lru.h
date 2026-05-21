// Small per-thread LRU cache machinery for kernel precomputes.
//
// Shape: callers declare a payload type that derives from `CacheKey`
// (`size`, `layout_sig`, `content_tag`) and hold a `std::array<Payload, N>`
// in TLS. `refresh_cache_lru` looks up by (size, layout_sig, content_tag),
// promotes hits to slot 0, and on a miss evicts the LRU slot and hands the
// caller the empty slot 0 to fill.
//
// `content_tag` is a sampled FNV-1a fingerprint — a low-cost content
// invalidation signal that catches buffer reuse / localized edits without
// scanning full payloads. Hazard documented in `content_tag`'s docstring.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

#include "common/fnv1a.h"

namespace pymdp_ffi {

// Cache key carried by every LRU payload. Pointer identity is intentionally
// not part of the key — JAX may move the same data or recycle an address
// for different data. Content_tag stays in the key even when a pointer
// repeats: buffer addresses can be recycled for different same-shaped data
// across calls.
struct CacheKey {
  int64_t  size        = 0;
  uint64_t layout_sig  = 0;
  uint64_t content_tag = 0;
};

// Default slot count for per-thread precompute LRUs. Eight slots cover the
// common batch_size=4 rollout path for both A and B without thrashing
// between batch elements, while keeping the cache tiny.
inline constexpr std::size_t kPrecomputeCacheSlots = 8;

// Content-tag sample count: short prefix + 8 strided samples.
inline constexpr int kContentTagSamples = 8;

// Low-cost content fingerprint for cache invalidation. It mixes a short
// prefix plus strided samples, which catches common buffer reuse /
// localized edits without scanning full payloads on every rollout step.
//
// Hazard: this is a sampled fingerprint, not a hash. A caller that mutates
// the underlying buffer in-place between calls without touching any of the
// sampled positions would get a false cache hit and read stale precompute.
// JAX arrays are immutable so the production call path is safe; FFI users
// that pass in-place-mutated raw buffers must treat each mutation as a new
// logical input (e.g. by copying) to keep the cache correct.
inline uint64_t content_tag(const float* data, int64_t size) {
  if (size <= 0) return 0;
  // Belt-and-suspenders against a (nullptr, size > 0) caller: every current
  // caller gates on buffer presence upstream (in.pA != nullptr, pA_present,
  // flags.use_utility, ...) but a future caller forwarding an optional
  // buffer with its declared size would segfault inside the std::memcpy
  // below. Treat null + positive-size identically to size <= 0.
  if (data == nullptr) return 0;
  uint64_t      h      = kFnvOffsetBasis;
  const int64_t prefix = std::min<int64_t>(size, 16);
  for (int64_t idx = 0; idx < prefix; ++idx) {
    uint32_t bits;
    std::memcpy(&bits, data + idx, sizeof(bits));
    h = fnv1a64_mix(h, static_cast<uint64_t>(bits));
  }
  const int64_t samples[kContentTagSamples] = {
      0, (size > 1) ? 1 : 0, size / 8, size / 4, size / 2, (size * 3) / 4, (size * 7) / 8, size - 1,
  };
  for (long long idx : samples) {
    if (idx < 0) idx = 0;
    if (idx >= size) idx = size - 1;
    uint32_t bits;
    std::memcpy(&bits, data + idx, sizeof(bits));
    h = fnv1a64_mix(h, static_cast<uint64_t>(bits));
  }
  return h;
}

// Refresh the LRU cache and return slot 0. On miss, `recompute` receives the
// destination slot so callers can fill layout-specific payload fields.
//
// `Cache` must derive from / contain a `CacheKey` (exposing `size`,
// `layout_sig`, `content_tag` fields). Slot 0 is most-recent after the call.
template <class Cache, std::size_t N, class Recompute>
inline Cache* refresh_cache_lru(std::array<Cache, N>& arr, const float* ptr, int64_t size, uint64_t sig,
                                Recompute&& recompute) {
  const uint64_t tag = content_tag(ptr, size);
  for (std::size_t i = 0; i < N; ++i) {
    if (arr[i].size == size && arr[i].layout_sig == sig && arr[i].content_tag == tag) {
      if (i != 0) {
        Cache hit = std::move(arr[i]);
        for (std::size_t j = i; j > 0; --j) arr[j] = std::move(arr[j - 1]);
        arr[0] = std::move(hit);
      }
      arr[0].content_tag = tag;
      return &arr[0];
    }
  }
  // Miss: evict the least-recent slot and shift the remaining entries back.
  for (std::size_t j = N - 1; j > 0; --j) arr[j] = std::move(arr[j - 1]);
  recompute(arr[0]);
  arr[0].size        = size;
  arr[0].layout_sig  = sig;
  arr[0].content_tag = tag;
  return &arr[0];
}

}  // namespace pymdp_ffi
