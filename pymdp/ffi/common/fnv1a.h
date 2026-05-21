// FNV-1a 64-bit hash + streaming mixer. Shared by every kernel that needs a
// cheap, header-only content/layout signature (fpi attr cache, neg_efe layout /
// content / pm-action sigs). No quality requirement beyond separating common
// layout/content changes with low overhead.

#pragma once

#include <cstddef>
#include <cstdint>

namespace pymdp_ffi {

inline constexpr uint64_t kFnvOffsetBasis = 0xcbf29ce484222325ULL;
inline constexpr uint64_t kFnvPrime       = 0x100000001b3ULL;

// Mix one 64-bit word into a running hash. Match `acc ^= x; acc *= prime;`.
inline uint64_t fnv1a64_mix(uint64_t acc, uint64_t x) {
  acc ^= x;
  acc *= kFnvPrime;
  return acc;
}

// Byte-wise FNV-1a over `[data, data+bytes)`. `seed` defaults to the FNV-1a
// offset basis; pass the previous return value to stream-hash multiple ranges.
inline uint64_t fnv1a64(const void* data, size_t bytes, uint64_t seed = kFnvOffsetBasis) {
  const uint8_t* p = static_cast<const uint8_t*>(data);
  uint64_t       h = seed;
  for (size_t i = 0; i < bytes; ++i) {
    h ^= p[i];
    h *= kFnvPrime;
  }
  return h;
}

}  // namespace pymdp_ffi
