// FPI convergence + per-factor softmax helpers.
//
// CPU-only; the CUDA FPI path implements its own softmax/convergence on
// device.

#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>

#include "common/kernel_primitives.h"

namespace pymdp_ffi {

// Per-factor softmax: copies src[off..off+S[f]] into dst and replaces in-
// place with stable softmax. Same shape signature as the per-iteration q[f]
// update and the final q_out write. src/dst are distinct buffers (scratch_
// log_q vs scratch_q in fpi_one_batch, scratch_log_q vs q_out at the tail),
// so the memcpy is non-overlapping and restrict applies.
inline void softmax_per_factor(int64_t F, const int64_t* __restrict__ S, const int64_t* __restrict__ lp_offsets,
                               const float* __restrict__ src, float* __restrict__ dst) {
  for (int64_t f = 0; f < F; ++f) {
    const int64_t n   = S[f];
    const int64_t off = lp_offsets[f];
    std::memcpy(dst + off, src + off, n * sizeof(float));
    softmax_inplace(dst + off, n);
  }
}

// Per-factor softmax + convergence snapshot. Identical to softmax_per_factor
// but additionally writes src[off..off+S[f]] into prev[off..off+S[f]] — the
// convergence check's log_q_prev. Compiles to a single src-load loop that
// fans out to both dst and prev, so the snapshot piggybacks on a load that
// the softmax pass would have done anyway. Folding the snapshot here means
// fpi_one_batch can drop the explicit `std::memcpy(scratch_log_q_prev,
// scratch_log_q, total_S * sizeof(float))` from the per-iter prelude — net
// save is one full-buffer pass over scratch_log_q per iter.
//
// src/dst/prev are all distinct: src = scratch_log_q, dst = scratch_q,
// prev = scratch_log_q_prev (three distinct FpiScratch vectors).
inline void softmax_per_factor_snapshot(int64_t F, const int64_t* __restrict__ S,
                                        const int64_t* __restrict__ lp_offsets, const float* __restrict__ src,
                                        float* __restrict__ dst, float* __restrict__ prev) {
  for (int64_t f = 0; f < F; ++f) {
    const int64_t n   = S[f];
    const int64_t off = lp_offsets[f];
    // Single-pass read of src fanning out to dst and prev. Autovectorizes
    // to a NEON/AVX2 load + two stores; same memory-bandwidth footprint as
    // the bare memcpy in softmax_per_factor since src is hot for the
    // duration.
    const float* s = src + off;
    float*       d = dst + off;
    float*       p = prev + off;
    for (int64_t i = 0; i < n; ++i) {
      const float v = s[i];
      d[i]          = v;
      p[i]          = v;
    }
    softmax_inplace(d, n);
  }
}

// max |a[i] - b[i]| over [0, n). Used by the FPI convergence-check path.
// Scalar is fine — total_S is small (~72 floats for production) and the
// kernel does at most num_iter passes. Auto-vectorizes to AVX2/NEON
// max-of-abs-diff.
inline float max_abs_diff(const float* __restrict__ a, const float* __restrict__ b, int64_t n) {
  float m = 0.0f;
  for (int64_t i = 0; i < n; ++i) {
    const float d = std::fabs(a[i] - b[i]);
    if (d > m) m = d;
  }
  return m;
}

// Hard-coded FPI early-stop tolerance: after each body iter, the kernel
// computes `max|log_q - log_q_prev|` and breaks once below this. Empirically
// chosen at 1e-5 — well below test_fpi_ffi.py's parity atol of 1e-6 (max
// observed |q - q_ref| was 2.7e-7 across all fixtures), still loose enough
// to fire on small/easy shapes (fpi_inference, fpi_high_rank). Larger
// problems (fpi_large) don't converge to this threshold within typical
// num_iter=16 and run the full loop. Cost when not firing: one memcpy +
// max-abs-diff per iter, ~5% of per-iter cost — recouped many times over
// when it does.
constexpr float kFpiConvergenceTol = 1e-5f;

}  // namespace pymdp_ffi
