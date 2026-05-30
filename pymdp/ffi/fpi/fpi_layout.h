// Layout metadata for the FPI FFI ABI: the five attribute spans XLA hands
// every FPI target, plus a content fingerprint (`fpi_layout_signature`)
// over those spans used to key the CUDA dispatch-table cache.
//
// Mirrors neg_efe/neg_efe_layout.h's `LayoutSpans` role but trimmed to the
// FPI surface — FPI has no per-modality C / I / B / qs_offsets, just the
// (S, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets) tuple.

#pragma once

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "common/fnv1a.h"

namespace pymdp_ffi {

// The five attribute spans XLA hands every FPI target. Field order matches
// the Bind chain in xla_register.cc and the JAX wrapper in pymdp/ffi/_fpi.py.
struct FpiSpans {
  FfiInt64Span S;
  FfiInt64Span ll_offsets;
  FfiInt64Span lp_offsets;
  FfiInt64Span A_dep_flat;
  FfiInt64Span A_dep_offsets;
};

// FNV-1a fingerprint over the five attr-span byte ranges plus an (F, M)
// shape tag. Used as the CUDA dispatch-table cache key in fpi_cuda_cache.
// Hashing the raw attrs skips both H2D uploads and validate_fpi_attrs +
// per-modality dispatch build on cache hits.
//
// `sig == 0` is reserved for "never uploaded yet" (initial cache state);
// a collision is treated as a miss — re-validate, re-build, re-upload.
inline uint64_t fpi_layout_signature(const FpiSpans& spans, int64_t F, int64_t M) {
  uint64_t sig = fnv1a64(spans.S.begin(), static_cast<size_t>(spans.S.size()) * sizeof(int64_t));
  sig          = fnv1a64(spans.ll_offsets.begin(), static_cast<size_t>(spans.ll_offsets.size()) * sizeof(int64_t), sig);
  sig          = fnv1a64(spans.lp_offsets.begin(), static_cast<size_t>(spans.lp_offsets.size()) * sizeof(int64_t), sig);
  sig          = fnv1a64(spans.A_dep_flat.begin(), static_cast<size_t>(spans.A_dep_flat.size()) * sizeof(int64_t), sig);
  sig = fnv1a64(spans.A_dep_offsets.begin(), static_cast<size_t>(spans.A_dep_offsets.size()) * sizeof(int64_t), sig);
  // Mix F and M in too — defends against pathological cases where two
  // distinct (F, M) shapes hash equivalently on their prefixes.
  const uint64_t shape_tag = (static_cast<uint64_t>(F) << 32) ^ static_cast<uint64_t>(M);
  sig                      = fnv1a64(&shape_tag, sizeof(shape_tag), sig);
  return sig;
}

}  // namespace pymdp_ffi
