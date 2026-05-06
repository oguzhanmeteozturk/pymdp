// Generic CPU kernel primitives reusable across pymdp FFI kernels (neg-EFE
// and FPI today).
//
// Header-only; mark everything `inline` so multiple includers don't ODR-clash.
//
// Contents:
//   logexp_f32.h (included below) — kLogEps, xlogx, entropy, softmax_inplace,
//                       NEON vexp/vlog when PYMDP_FFI_VEC_LOGEXP.
//   sgemv_rm_f32      — row-major sgemv; AArch64 NEON, scalar fallback elsewhere.
//   sgemv_rm_compact  — compact-input sgemv; ARMv8.2 FP16/FML path when available,
//                       f32 fallback elsewhere.
//   sdot_f32          — dot product; AArch64 NEON, scalar fallback elsewhere.
//   pack_f32_to_f16   — f32→compact conversion for the EFE A path.
//   KernelTimer       — RAII profiler gated on an env var; logs on dtor.
//

#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "cpu_capabilities.h"
#include "omp_helpers.h"

// Make the intrinsics dependency explicit instead of relying on
// cpu_capabilities.h pulling these in. Each guarded include is a no-op
// when the matching ISA path is disabled.
#if PYMDP_FFI_HAS_X86_AVX2 || PYMDP_FFI_HAS_X86_AVX512F
#include <immintrin.h>
#endif
#if PYMDP_FFI_HAS_AARCH64_NEON
#include <arm_neon.h>
#endif

namespace pymdp_ffi {

inline bool is_nonfinite_f32(float x) noexcept {
  uint32_t bits;
  std::memcpy(&bits, &x, sizeof(bits));
  return (bits & 0x7f800000u) == 0x7f800000u;
}

#include "logexp_f32.h"

// f32 digamma (psi) for x > 0. Used by spm_wnorm in the param-info-gain path.
//
// Recurrence to shift x >= 6, then asymptotic Bernoulli expansion:
//   psi(x) ≈ log(x) - 1/(2x) - 1/(12 x^2) + 1/(120 x^4) - 1/(252 x^6) + 1/(240 x^8)
// Accuracy at x >= 6 is ~1e-7 (saturates f32). Below 6 the recurrence
// `psi(x) = psi(x+1) - 1/x` brings us into the asymptotic regime.
//
// pymdp clips Dirichlet inputs to MINVAL = float_eps (~1.2e-7) before the
// digamma call. Because the recurrence increments x by 1 each step, even
// the smallest clipped input reaches x >= 6 in at most 6 iterations. In
// steady state pA/pB are concentration parameters initialized to ~1 and
// only grow with learning, so the loop typically runs ~5 times.
inline float digamma_f32(float x) noexcept {
  // -ffast-math poisons std::numeric_limits<float>::infinity(); use a large
  // negative sentinel instead. The early return is defensive — callers (the
  // wA / wB precompute) always mask with (p > 0) before invoking digamma, so
  // this branch should never fire on the hot path.
  if (!(x > 0.0f) || is_nonfinite_f32(x)) return -1e30f;
  float result = 0.0f;
  // Cap at 64 recurrence steps. With x += 1 per iteration, this can only
  // bind for x < -58, which the early return above already rejects. The
  // cap is purely belt-and-braces against future refactors that might
  // weaken the entry guard.
  int steps = 0;
  while (x < 6.0f && steps < 64) {
    result -= 1.0f / x;
    x += 1.0f;
    ++steps;
  }
  const float x2 = x * x;
  const float x4 = x2 * x2;
  const float x6 = x4 * x2;
  const float x8 = x4 * x4;
  result +=
      std::log(x) - 0.5f / x - 1.0f / (12.0f * x2) + 1.0f / (120.0f * x4) - 1.0f / (252.0f * x6) + 1.0f / (240.0f * x8);
  return result;
}

#if PYMDP_FFI_HAS_X86_AVX2
// Horizontal sum of an AVX2 8-wide f32 lane. Shared by sgemv_rm_f32 and sdot_f32.
inline float hsum256_f32(__m256 v) noexcept {
  __m128 lo = _mm256_castps256_ps128(v);
  __m128 hi = _mm256_extractf128_ps(v, 1);
  __m128 s  = _mm_add_ps(lo, hi);
  s         = _mm_hadd_ps(s, s);
  s         = _mm_hadd_ps(s, s);
  return _mm_cvtss_f32(s);
}
#endif
#if PYMDP_FFI_HAS_X86_AVX512F
// AVX-512 16-wide horizontal sum. Shared by sgemv_rm_f32 and sdot_f32.
inline float hsum512_f32(__m512 v) noexcept {
  return _mm512_reduce_add_ps(v);
}
#endif

// Row-major sgemv: y[i] = sum_j A[i*lda + j] * x[j], for i in [0, m).
//
// 4-row × wide-col unroll gives 4 independent accumulator chains and amortizes
// the load of x across 4 rows. AArch64 NEON uses 4×4 (vfmaq_f32); AVX-512F uses
// 4×16 (_mm512_fmadd_ps); AVX2 uses 4×8 (_mm256_fmadd_ps). The `ACCUM` template
// flag switches the write step between `y[i] = s` (sgemv_rm_f32) and
// `y_add[i] += s` (sgemv_rm_f32_add) — `if constexpr` eliminates the dead arm
// at compile time so each instantiation generates the same code the explicit
// non-templated forms used to produce.
//
// Caller contract: A, x, y/y_add are pairwise non-overlapping buffers —
// restrict lets the compiler hoist loads of x[j] across the 4-row unroll
// instead of reloading after each write.
namespace detail {
template <bool ACCUM>
inline void sgemv_rm_f32_impl(int64_t m, int64_t n, const float* __restrict__ A, int64_t lda,
                              const float* __restrict__ x, float* __restrict__ y) noexcept {
  auto write4 = [&](int64_t i, float s0, float s1, float s2, float s3) {
    if constexpr (ACCUM) {
      y[i + 0] += s0;
      y[i + 1] += s1;
      y[i + 2] += s2;
      y[i + 3] += s3;
    } else {
      y[i + 0] = s0;
      y[i + 1] = s1;
      y[i + 2] = s2;
      y[i + 3] = s3;
    }
  };
  auto write1 = [&](int64_t i, float s) {
    if constexpr (ACCUM) {
      y[i] += s;
    } else {
      y[i] = s;
    }
  };

  if (m <= 0 || n <= 0) return;
#if PYMDP_FFI_HAS_AARCH64_NEON
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    const float* r0   = A + (i + 0) * lda;
    const float* r1   = A + (i + 1) * lda;
    const float* r2   = A + (i + 2) * lda;
    const float* r3   = A + (i + 3) * lda;
    float32x4_t  acc0 = vdupq_n_f32(0.0f);
    float32x4_t  acc1 = vdupq_n_f32(0.0f);
    float32x4_t  acc2 = vdupq_n_f32(0.0f);
    float32x4_t  acc3 = vdupq_n_f32(0.0f);
    int64_t      j    = 0;
    for (; j + 4 <= n; j += 4) {
      float32x4_t xv = vld1q_f32(x + j);
      acc0           = vfmaq_f32(acc0, vld1q_f32(r0 + j), xv);
      acc1           = vfmaq_f32(acc1, vld1q_f32(r1 + j), xv);
      acc2           = vfmaq_f32(acc2, vld1q_f32(r2 + j), xv);
      acc3           = vfmaq_f32(acc3, vld1q_f32(r3 + j), xv);
    }
    float s0 = vaddvq_f32(acc0);
    float s1 = vaddvq_f32(acc1);
    float s2 = vaddvq_f32(acc2);
    float s3 = vaddvq_f32(acc3);
    for (; j < n; ++j) {
      float xj = x[j];
      s0 += r0[j] * xj;
      s1 += r1[j] * xj;
      s2 += r2[j] * xj;
      s3 += r3[j] * xj;
    }
    write4(i, s0, s1, s2, s3);
  }
  for (; i < m; ++i) {
    const float* r   = A + i * lda;
    float32x4_t  acc = vdupq_n_f32(0.0f);
    int64_t      j   = 0;
    for (; j + 4 <= n; j += 4) {
      acc = vfmaq_f32(acc, vld1q_f32(r + j), vld1q_f32(x + j));
    }
    float s = vaddvq_f32(acc);
    for (; j < n; ++j) s += r[j] * x[j];
    write1(i, s);
  }
#elif PYMDP_FFI_HAS_X86_AVX512F
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    const float* r0   = A + (i + 0) * lda;
    const float* r1   = A + (i + 1) * lda;
    const float* r2   = A + (i + 2) * lda;
    const float* r3   = A + (i + 3) * lda;
    __m512       acc0 = _mm512_setzero_ps();
    __m512       acc1 = _mm512_setzero_ps();
    __m512       acc2 = _mm512_setzero_ps();
    __m512       acc3 = _mm512_setzero_ps();
    int64_t      j    = 0;
    for (; j + 16 <= n; j += 16) {
      __m512 xv = _mm512_loadu_ps(x + j);
      acc0      = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + j), xv, acc0);
      acc1      = _mm512_fmadd_ps(_mm512_loadu_ps(r1 + j), xv, acc1);
      acc2      = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + j), xv, acc2);
      acc3      = _mm512_fmadd_ps(_mm512_loadu_ps(r3 + j), xv, acc3);
    }
    float s0 = hsum512_f32(acc0);
    float s1 = hsum512_f32(acc1);
    float s2 = hsum512_f32(acc2);
    float s3 = hsum512_f32(acc3);
    for (; j < n; ++j) {
      float xj = x[j];
      s0 += r0[j] * xj;
      s1 += r1[j] * xj;
      s2 += r2[j] * xj;
      s3 += r3[j] * xj;
    }
    write4(i, s0, s1, s2, s3);
  }
  for (; i < m; ++i) {
    const float* r   = A + i * lda;
    __m512       acc = _mm512_setzero_ps();
    int64_t      j   = 0;
    for (; j + 16 <= n; j += 16) {
      acc = _mm512_fmadd_ps(_mm512_loadu_ps(r + j), _mm512_loadu_ps(x + j), acc);
    }
    float s = hsum512_f32(acc);
    for (; j < n; ++j) s += r[j] * x[j];
    write1(i, s);
  }
#elif PYMDP_FFI_HAS_X86_AVX2
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    const float* r0   = A + (i + 0) * lda;
    const float* r1   = A + (i + 1) * lda;
    const float* r2   = A + (i + 2) * lda;
    const float* r3   = A + (i + 3) * lda;
    __m256       acc0 = _mm256_setzero_ps();
    __m256       acc1 = _mm256_setzero_ps();
    __m256       acc2 = _mm256_setzero_ps();
    __m256       acc3 = _mm256_setzero_ps();
    int64_t      j    = 0;
    for (; j + 8 <= n; j += 8) {
      __m256 xv = _mm256_loadu_ps(x + j);
      acc0      = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + j), xv, acc0);
      acc1      = _mm256_fmadd_ps(_mm256_loadu_ps(r1 + j), xv, acc1);
      acc2      = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + j), xv, acc2);
      acc3      = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + j), xv, acc3);
    }
    float s0 = hsum256_f32(acc0);
    float s1 = hsum256_f32(acc1);
    float s2 = hsum256_f32(acc2);
    float s3 = hsum256_f32(acc3);
    for (; j < n; ++j) {
      float xj = x[j];
      s0 += r0[j] * xj;
      s1 += r1[j] * xj;
      s2 += r2[j] * xj;
      s3 += r3[j] * xj;
    }
    write4(i, s0, s1, s2, s3);
  }
  for (; i < m; ++i) {
    const float* r   = A + i * lda;
    __m256       acc = _mm256_setzero_ps();
    int64_t      j   = 0;
    for (; j + 8 <= n; j += 8) {
      acc = _mm256_fmadd_ps(_mm256_loadu_ps(r + j), _mm256_loadu_ps(x + j), acc);
    }
    float s = hsum256_f32(acc);
    for (; j < n; ++j) s += r[j] * x[j];
    write1(i, s);
  }
#else
  for (int64_t i = 0; i < m; ++i) {
    const float* r = A + i * lda;
    float        s = 0.0f;
    for (int64_t j = 0; j < n; ++j) s += r[j] * x[j];
    write1(i, s);
  }
#endif
}
}  // namespace detail

inline void sgemv_rm_f32(int64_t m, int64_t n, const float* __restrict__ A, int64_t lda, const float* __restrict__ x,
                         float* __restrict__ y) noexcept {
  detail::sgemv_rm_f32_impl</*ACCUM=*/false>(m, n, A, lda, x, y);
}

// Accumulating weighted leading-dim reduction:
//   out_add[k] += sum_{i in [0, lead)} w[i] * mat[i*inner + k]
// Caller is responsible for the initial value of out_add.
//
// 4-row outer unroll: reads out_add[k:k+VEC] once and accumulates four
// weighted FMAs before storing back, cutting L1 round-trips on out_add by ~4×
// vs a per-row pattern. Tail rows fall through to a 1-row SIMD-inner loop.
// NEON: 4×4-wide. AVX-512F: 4×16-wide. AVX2: 4×8-wide. Scalar fallback.
//
// The inner FMAs are split into two independent 2-deep chains (rows {0,1}
// and {2,3}) joined by a final vadd. A naive single-accumulator chain has
// 4 dependent FMAs on the critical path; splitting halves it. The win
// scales with FMA latency: ~1.5× critical-path cycles on Cortex-A57
// (FMA lat ~9c, single FP issue port) and ~1.2-1.3× on Apple silicon /
// modern x86 (FMA lat ~3-5c, multiple FP ports — gain comes from freeing
// FP issue slots for OoO across iterations rather than latency).
inline void axpy_fold_leading_add(int64_t lead, int64_t inner, const float* __restrict__ mat,
                                  const float* __restrict__ w, float* __restrict__ out_add) noexcept {
  if (lead <= 0 || inner <= 0) return;
#if PYMDP_FFI_HAS_AARCH64_NEON
  int64_t i = 0;
  for (; i + 4 <= lead; i += 4) {
    const float* r0 = mat + (i + 0) * inner;
    const float* r1 = mat + (i + 1) * inner;
    const float* r2 = mat + (i + 2) * inner;
    const float* r3 = mat + (i + 3) * inner;
    float32x4_t  w0 = vdupq_n_f32(w[i + 0]);
    float32x4_t  w1 = vdupq_n_f32(w[i + 1]);
    float32x4_t  w2 = vdupq_n_f32(w[i + 2]);
    float32x4_t  w3 = vdupq_n_f32(w[i + 3]);
    int64_t      k  = 0;
    for (; k + 4 <= inner; k += 4) {
      // Two independent 2-deep FMA chains: a0={r0,r1}, a1={r2,r3}.
      // Folds the existing out_add[k] load into chain a0 as the seed,
      // and starts a1 from r1*w1 via vmul. Final vadd joins.
      float32x4_t v  = vld1q_f32(out_add + k);
      float32x4_t a0 = vfmaq_f32(v, vld1q_f32(r0 + k), w0);
      float32x4_t a1 = vmulq_f32(vld1q_f32(r1 + k), w1);
      a0             = vfmaq_f32(a0, vld1q_f32(r2 + k), w2);
      a1             = vfmaq_f32(a1, vld1q_f32(r3 + k), w3);
      vst1q_f32(out_add + k, vaddq_f32(a0, a1));
    }
    for (; k < inner; ++k) {
      out_add[k] += w[i + 0] * r0[k] + w[i + 1] * r1[k] + w[i + 2] * r2[k] + w[i + 3] * r3[k];
    }
  }
  for (; i < lead; ++i) {
    const float  wi  = w[i];
    const float* row = mat + i * inner;
    float32x4_t  wv  = vdupq_n_f32(wi);
    int64_t      k   = 0;
    for (; k + 4 <= inner; k += 4) {
      float32x4_t v = vld1q_f32(out_add + k);
      v             = vfmaq_f32(v, vld1q_f32(row + k), wv);
      vst1q_f32(out_add + k, v);
    }
    for (; k < inner; ++k) out_add[k] += wi * row[k];
  }
#elif PYMDP_FFI_HAS_X86_AVX512F
  int64_t i = 0;
  for (; i + 4 <= lead; i += 4) {
    const float* r0 = mat + (i + 0) * inner;
    const float* r1 = mat + (i + 1) * inner;
    const float* r2 = mat + (i + 2) * inner;
    const float* r3 = mat + (i + 3) * inner;
    __m512       w0 = _mm512_set1_ps(w[i + 0]);
    __m512       w1 = _mm512_set1_ps(w[i + 1]);
    __m512       w2 = _mm512_set1_ps(w[i + 2]);
    __m512       w3 = _mm512_set1_ps(w[i + 3]);
    int64_t      k  = 0;
    for (; k + 16 <= inner; k += 16) {
      // Two independent 2-deep FMA chains; see NEON branch for rationale.
      __m512 v  = _mm512_loadu_ps(out_add + k);
      __m512 a0 = _mm512_fmadd_ps(_mm512_loadu_ps(r0 + k), w0, v);
      __m512 a1 = _mm512_mul_ps(_mm512_loadu_ps(r1 + k), w1);
      a0        = _mm512_fmadd_ps(_mm512_loadu_ps(r2 + k), w2, a0);
      a1        = _mm512_fmadd_ps(_mm512_loadu_ps(r3 + k), w3, a1);
      _mm512_storeu_ps(out_add + k, _mm512_add_ps(a0, a1));
    }
    for (; k < inner; ++k) {
      out_add[k] += w[i + 0] * r0[k] + w[i + 1] * r1[k] + w[i + 2] * r2[k] + w[i + 3] * r3[k];
    }
  }
  for (; i < lead; ++i) {
    const float  wi  = w[i];
    const float* row = mat + i * inner;
    __m512       wv  = _mm512_set1_ps(wi);
    int64_t      k   = 0;
    for (; k + 16 <= inner; k += 16) {
      __m512 v = _mm512_loadu_ps(out_add + k);
      v        = _mm512_fmadd_ps(_mm512_loadu_ps(row + k), wv, v);
      _mm512_storeu_ps(out_add + k, v);
    }
    for (; k < inner; ++k) out_add[k] += wi * row[k];
  }
#elif PYMDP_FFI_HAS_X86_AVX2
  int64_t i = 0;
  for (; i + 4 <= lead; i += 4) {
    const float* r0 = mat + (i + 0) * inner;
    const float* r1 = mat + (i + 1) * inner;
    const float* r2 = mat + (i + 2) * inner;
    const float* r3 = mat + (i + 3) * inner;
    __m256       w0 = _mm256_set1_ps(w[i + 0]);
    __m256       w1 = _mm256_set1_ps(w[i + 1]);
    __m256       w2 = _mm256_set1_ps(w[i + 2]);
    __m256       w3 = _mm256_set1_ps(w[i + 3]);
    int64_t      k  = 0;
    for (; k + 8 <= inner; k += 8) {
      // Two independent 2-deep FMA chains; see NEON branch for rationale.
      __m256 v  = _mm256_loadu_ps(out_add + k);
      __m256 a0 = _mm256_fmadd_ps(_mm256_loadu_ps(r0 + k), w0, v);
      __m256 a1 = _mm256_mul_ps(_mm256_loadu_ps(r1 + k), w1);
      a0        = _mm256_fmadd_ps(_mm256_loadu_ps(r2 + k), w2, a0);
      a1        = _mm256_fmadd_ps(_mm256_loadu_ps(r3 + k), w3, a1);
      _mm256_storeu_ps(out_add + k, _mm256_add_ps(a0, a1));
    }
    for (; k < inner; ++k) {
      out_add[k] += w[i + 0] * r0[k] + w[i + 1] * r1[k] + w[i + 2] * r2[k] + w[i + 3] * r3[k];
    }
  }
  for (; i < lead; ++i) {
    const float  wi  = w[i];
    const float* row = mat + i * inner;
    __m256       wv  = _mm256_set1_ps(wi);
    int64_t      k   = 0;
    for (; k + 8 <= inner; k += 8) {
      __m256 v = _mm256_loadu_ps(out_add + k);
      v        = _mm256_fmadd_ps(_mm256_loadu_ps(row + k), wv, v);
      _mm256_storeu_ps(out_add + k, v);
    }
    for (; k < inner; ++k) out_add[k] += wi * row[k];
  }
#else
  for (int64_t i = 0; i < lead; ++i) {
    const float  wi  = w[i];
    const float* row = mat + i * inner;
    for (int64_t k = 0; k < inner; ++k) out_add[k] += wi * row[k];
  }
#endif
}

// Non-accumulating variant. For inner sizes ~16–400 floats the memset cost
// is negligible and the SIMD path runs against a hot L1 line, so this
// delegates to the accumulating form after zeroing.
inline void axpy_fold_leading(int64_t lead, int64_t inner, const float* __restrict__ mat, const float* __restrict__ w,
                              float* __restrict__ out) noexcept {
  if (lead <= 0 || inner <= 0) return;
  std::memset(out, 0, static_cast<size_t>(inner) * sizeof(float));
  axpy_fold_leading_add(lead, inner, mat, w, out);
}

// Accumulating variant of sgemv_rm_f32: y_add[i] += sum_j A[i*lda+j] * x[j].
inline void sgemv_rm_f32_add(int64_t m, int64_t n, const float* __restrict__ A, int64_t lda,
                             const float* __restrict__ x, float* __restrict__ y_add) noexcept {
  detail::sgemv_rm_f32_impl</*ACCUM=*/true>(m, n, A, lda, x, y_add);
}

// Final Kronecker step: out[c, s] = cur[c] * q_last[s], cast-stored to OutT.
// OutT == float writes f32; OutT == FfiF16 stores f16 only when FP16/FML is
// available and f32 otherwise.
template <class OutT>
inline void kron_final_outer(int64_t cur_k, const float* cur, int64_t S_last, const float* q_last, OutT* out) noexcept {
  for (int64_t c = 0; c < cur_k; ++c) {
    const float a   = cur[c];
    OutT*       dst = out + c * S_last;
    for (int64_t s = 0; s < S_last; ++s) {
      dst[s] = static_cast<OutT>(a * q_last[s]);
    }
  }
}

// Pack `Bn` consecutive slices of `base` into a contiguous `[Bn, slice_size]`
// buffer. `dst` is resized as needed. Each slice is copied from
// `base + b * batch_stride + slice_offset`. Used by the CUDA host-side
// upload paths to stage one modality / factor / timestep into a single buffer
// before handing it to the GPU runtime.
inline void pack_batched_slices(std::vector<float>* dst, const float* base, int Bn, int64_t batch_stride,
                                int64_t slice_offset, size_t slice_size) {
  if (Bn <= 0 || slice_size == 0) {
    if (!dst->empty()) dst->resize(0);
    return;
  }
  const size_t total = static_cast<size_t>(Bn) * slice_size;
  if (dst->size() != total) dst->resize(total);
  for (int b = 0; b < Bn; ++b) {
    std::memcpy(dst->data() + static_cast<size_t>(b) * slice_size,
                base + static_cast<int64_t>(b) * batch_stride + slice_offset, slice_size * sizeof(float));
  }
}

// Pack n f32 values into the EFE kernel's compact storage.
//
// ARMv8.2 FP16/FML build: NEON f32→f16 conversion (vcvt_f16_f32).
// ARMv8.0 / non-aarch64 build: FfiF16 == float, so this is a contiguous
// f32→f32 copy — memcpy beats a scalar element-wise loop and lets the
// libc memcpy use whatever vector ISA is available.
inline void pack_f32_to_f16(int64_t n, const float* src, FfiF16* dst) noexcept {
  if (n <= 0) return;
#if PYMDP_FFI_HAS_F16_FML
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    vst1_f16(dst + j, vcvt_f16_f32(vld1q_f32(src + j)));
  }
  for (; j < n; ++j) dst[j] = static_cast<FfiF16>(src[j]);
#else
  static_assert(std::is_same<FfiF16, float>::value, "fallback assumes f32 storage");
  std::memcpy(dst, src, static_cast<size_t>(n) * sizeof(float));
#endif
}

// Row-major sgemv over compact inputs with f32 accumulation:
//   y[i] = sum_j (float)A[i*lda+j] * (float)x[j]
// ARMv8.2 FP16/FML uses f16*f16->f32 instructions; fallback builds (FfiF16
// aliased to float) reuse sgemv_rm_f32 via reinterpret_cast — same math,
// and inherits whatever SIMD backend sgemv_rm_f32 selects: NEON-f32 on
// ARMv8.0-A (Cortex-A57), AVX2/FMA or AVX-512F on x86,
// scalar only when no SIMD ISA is available.
inline void sgemv_rm_compact(int64_t m, int64_t n, const FfiF16* A, int64_t lda, const FfiF16* x, float* y) noexcept {
  if (m <= 0 || n <= 0) return;
#if PYMDP_FFI_HAS_F16_FML
  int64_t i = 0;
  for (; i + 4 <= m; i += 4) {
    const FfiF16* r0 = A + (i + 0) * lda;
    const FfiF16* r1 = A + (i + 1) * lda;
    const FfiF16* r2 = A + (i + 2) * lda;
    const FfiF16* r3 = A + (i + 3) * lda;
    // Split low/high accumulators so vfmlalq_low and vfmlalq_high don't
    // serialize on the same destination register -- the low/high pair of a
    // single f16x8 load becomes two independent FMA chains.
    float32x4_t a0l = vdupq_n_f32(0.0f), a0h = vdupq_n_f32(0.0f);
    float32x4_t a1l = vdupq_n_f32(0.0f), a1h = vdupq_n_f32(0.0f);
    float32x4_t a2l = vdupq_n_f32(0.0f), a2h = vdupq_n_f32(0.0f);
    float32x4_t a3l = vdupq_n_f32(0.0f), a3h = vdupq_n_f32(0.0f);
    int64_t     j = 0;
    for (; j + 8 <= n; j += 8) {
      float16x8_t xv  = vld1q_f16(x + j);
      float16x8_t rv0 = vld1q_f16(r0 + j);
      float16x8_t rv1 = vld1q_f16(r1 + j);
      float16x8_t rv2 = vld1q_f16(r2 + j);
      float16x8_t rv3 = vld1q_f16(r3 + j);
      a0l             = vfmlalq_low_f16(a0l, rv0, xv);
      a0h             = vfmlalq_high_f16(a0h, rv0, xv);
      a1l             = vfmlalq_low_f16(a1l, rv1, xv);
      a1h             = vfmlalq_high_f16(a1h, rv1, xv);
      a2l             = vfmlalq_low_f16(a2l, rv2, xv);
      a2h             = vfmlalq_high_f16(a2h, rv2, xv);
      a3l             = vfmlalq_low_f16(a3l, rv3, xv);
      a3h             = vfmlalq_high_f16(a3h, rv3, xv);
    }
    float s0 = vaddvq_f32(vaddq_f32(a0l, a0h));
    float s1 = vaddvq_f32(vaddq_f32(a1l, a1h));
    float s2 = vaddvq_f32(vaddq_f32(a2l, a2h));
    float s3 = vaddvq_f32(vaddq_f32(a3l, a3h));
    for (; j < n; ++j) {
      float xj = static_cast<float>(x[j]);
      s0 += static_cast<float>(r0[j]) * xj;
      s1 += static_cast<float>(r1[j]) * xj;
      s2 += static_cast<float>(r2[j]) * xj;
      s3 += static_cast<float>(r3[j]) * xj;
    }
    y[i + 0] = s0;
    y[i + 1] = s1;
    y[i + 2] = s2;
    y[i + 3] = s3;
  }
  for (; i < m; ++i) {
    const FfiF16* r    = A + i * lda;
    float32x4_t   accl = vdupq_n_f32(0.0f);
    float32x4_t   acch = vdupq_n_f32(0.0f);
    int64_t       j    = 0;
    for (; j + 8 <= n; j += 8) {
      float16x8_t rv = vld1q_f16(r + j);
      float16x8_t xv = vld1q_f16(x + j);
      accl           = vfmlalq_low_f16(accl, rv, xv);
      acch           = vfmlalq_high_f16(acch, rv, xv);
    }
    float s = vaddvq_f32(vaddq_f32(accl, acch));
    for (; j < n; ++j) s += static_cast<float>(r[j]) * static_cast<float>(x[j]);
    y[i] = s;
  }
#else
  // FfiF16 == float in this branch (ARMv8.0 / non-aarch64). On v8.0 aarch64
  // we still want the NEON-f32 4×4 unroll from sgemv_rm_f32; the cast is
  // safe because static_assert below confirms FfiF16 is float, so the
  // pointers and strides are bit-identical.
  static_assert(std::is_same<FfiF16, float>::value, "fallback assumes f32 storage");
  sgemv_rm_f32(m, n, reinterpret_cast<const float*>(A), lda, reinterpret_cast<const float*>(x), y);
#endif
}

// Dot product: sum_j x[j] * y[j]. Dual-accumulator inner loop on every SIMD
// path so two FMA chains run in parallel rather than serializing on a single
// register. NEON: 2×4-wide. AVX2: 2×8-wide. AVX-512: 2×16-wide.
inline float sdot_f32(int64_t n, const float* __restrict__ x, const float* __restrict__ y) noexcept {
  if (n <= 0) return 0.0f;
#if PYMDP_FFI_HAS_AARCH64_NEON
  float32x4_t acc0 = vdupq_n_f32(0.0f);
  float32x4_t acc1 = vdupq_n_f32(0.0f);
  int64_t     j    = 0;
  for (; j + 8 <= n; j += 8) {
    acc0 = vfmaq_f32(acc0, vld1q_f32(x + j), vld1q_f32(y + j));
    acc1 = vfmaq_f32(acc1, vld1q_f32(x + j + 4), vld1q_f32(y + j + 4));
  }
  float s = vaddvq_f32(vaddq_f32(acc0, acc1));
  for (; j < n; ++j) s += x[j] * y[j];
  return s;
#elif PYMDP_FFI_HAS_X86_AVX512F
  __m512  acc0 = _mm512_setzero_ps();
  __m512  acc1 = _mm512_setzero_ps();
  int64_t j    = 0;
  for (; j + 32 <= n; j += 32) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + j), _mm512_loadu_ps(y + j), acc0);
    acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(x + j + 16), _mm512_loadu_ps(y + j + 16), acc1);
  }
  for (; j + 16 <= n; j += 16) {
    acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(x + j), _mm512_loadu_ps(y + j), acc0);
  }
  float s = hsum512_f32(_mm512_add_ps(acc0, acc1));
  for (; j < n; ++j) s += x[j] * y[j];
  return s;
#elif PYMDP_FFI_HAS_X86_AVX2
  __m256  acc0 = _mm256_setzero_ps();
  __m256  acc1 = _mm256_setzero_ps();
  int64_t j    = 0;
  for (; j + 16 <= n; j += 16) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(y + j), acc0);
    acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j + 8), _mm256_loadu_ps(y + j + 8), acc1);
  }
  for (; j + 8 <= n; j += 8) {
    acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + j), _mm256_loadu_ps(y + j), acc0);
  }
  float s = hsum256_f32(_mm256_add_ps(acc0, acc1));
  for (; j < n; ++j) s += x[j] * y[j];
  return s;
#else
  float s = 0.0f;
  for (int64_t j = 0; j < n; ++j) s += x[j] * y[j];
  return s;
#endif
}

// Grow-only resize. vector::resize would shrink, which destroys elements
// beyond the new size; FFI scratch buffers use max-of-shapes-seen sizing,
// so growing is the only direction intended. Takes int64_t (matching the
// kernels' native size type) so callers don't decorate every call site
// with `static_cast<size_t>(...)`; non-positive n is treated as no-op.
template <class T> inline void ensure_at_least(std::vector<T>& v, int64_t n) {
  if (n <= 0) return;
  const size_t target = static_cast<size_t>(n);
  if (v.size() < target) v.resize(target);
}

// PYMDP_FFI_TIME=1 enables KernelTimer logs. The env var is read exactly once
// per process (via the function-static initializer) so the steady-state cost
// when unset is one atomic-bool load. Mutating PYMDP_FFI_TIME inside a running
// process will not toggle logging — set it before importing pymdp.ffi.
inline bool kernel_timer_enabled() noexcept {
  static const bool enabled = []() {
    const char* env = std::getenv("PYMDP_FFI_TIME");
    return env && std::strcmp(env, "1") == 0;
  }();
  return enabled;
}

// RAII timer. On destruction, if PYMDP_FFI_TIME=1, invokes the user-supplied
// log callable with elapsed microseconds. Templated on the callable to avoid
// std::function allocation.
template <class LogFn> class KernelTimer {
public:
  explicit KernelTimer(LogFn log) : start_(std::chrono::high_resolution_clock::now()), log_(std::move(log)) {}
  ~KernelTimer() noexcept {
    if (kernel_timer_enabled()) {
      const auto   end = std::chrono::high_resolution_clock::now();
      const double us  = std::chrono::duration<double, std::micro>(end - start_).count();
      // Swallow exceptions: throwing from a destructor during stack
      // unwinding terminates the process, and a profiling-only log call
      // is never worth that.
      try {
        log_(us);
      } catch (...) {  // NOLINT(bugprone-empty-catch): see comment above
      }
    }
  }
  KernelTimer(const KernelTimer&)            = delete;
  KernelTimer& operator=(const KernelTimer&) = delete;

private:
  std::chrono::high_resolution_clock::time_point start_;
  LogFn                                          log_;
};
template <class LogFn> KernelTimer(LogFn) -> KernelTimer<LogFn>;

}  // namespace pymdp_ffi
