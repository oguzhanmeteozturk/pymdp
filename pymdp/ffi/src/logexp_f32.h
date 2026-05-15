// Stable f32 log/exp helpers for pymdp FFI: NEON polynomial approximations on
// ARMv8.0 (Cortex-A57), scalar libm elsewhere via PYMDP_FFI_VEC_LOGEXP;
// entropy(q,n) and softmax_inplace(x,n) live here too.
//
// Include only from kernel_primitives.h: after global <arm_neon.h> (when
// AArch64) and inside namespace pymdp_ffi after is_nonfinite_f32.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "cpu_capabilities.h"

inline constexpr float kLogEps = 1e-12f;
// Smallest normal f32 bits (IEEE 754); used for SIMD log range reduction clamps.
inline constexpr uint32_t kF32MinNormalBits = 0x00800000u;

// Use (logf)/(expf): C float libm in global namespace (old gcc libstdc++ lacks std::logf/std::expf).
// Parentheses prevent accidental macro expansion where libm defines them as macros.
inline float xlogx(float x) noexcept {
  return x * (logf)(std::max(x, kLogEps));
}

// Degenerate softmax fallback: every element set to the same value (e.g. 1/n).
inline void fill_f32_inplace(float* x, int64_t n, float value) noexcept {
  for (int64_t i = 0; i < n; ++i) x[i] = value;
}

// Gate the polynomial expf/logf path on ARMv8.0 hosts (Cortex-A57),
// where libm is genuinely slow. Apple silicon (and any v8.2+ FP16/
// FML-capable host) has fast HW-assisted scalar expf/logf and regresses
// when the poly path is unconditional. The
// FML capability bit is the same signal we use to discriminate v8.0 from
// v8.2 elsewhere; flipping it via -DPYMDP_FFI_DISABLE_F16=1 forces the
// poly path on a v8.2 dev box for testing.
#if PYMDP_FFI_HAS_AARCH64_NEON && !PYMDP_FFI_HAS_F16_FML
#define PYMDP_FFI_VEC_LOGEXP 1
#else
#define PYMDP_FFI_VEC_LOGEXP 0
#endif

#if PYMDP_FFI_VEC_LOGEXP
// <arm_neon.h> is included globally by kernel_primitives.h before this
// namespace; do not include it here (nested include breaks Neon typedefs).
// Polynomial approximations of expf / logf for f32x4. Sse_mathfun-derived
// polynomial fits; close to libm for inference ranges, but not a
// correctly-rounded libm replacement. Caller contracts:
//   vexp_neon_f32: any input; clamped to [-88.376, 88.376] internally.
//   vlog_neon_f32: input must be > 0; result undefined for x <= 0.
//
// On Cortex-A57 (ARMv8.0) FP latency dominates: each vector is a long
// FMA/Horner chain with little ILP inside one lane. vexp2_neon_f32 /
// vlog_interleaved_neon_f32 runs two float32x4 lanes in lockstep (natural
// exp / ln respectively) so the scheduler can overlap latency across lanes.

struct f32x4x2 {
  float32x4_t a;
  float32x4_t b;
};

// Highest-degree polynomial terms (kVexpPolyY0 = 1.9875691500e-4f for x^7,
// kVlogPolyY0 = 7.0376836292e-2f for x^11) were dropped after measuring
// worst-case absolute error vs libm at the range-reduction edges:
// vexp ~1e-7 at |r|=log(2)/2, vlog ~5e-6 at |x|=0.414. Both are well
// inside jnp.allclose default rtol=1e-5 / atol=1e-8 and verified against
// the FFI parity suite. Saves one Horner step + one vdupq per call.
// Cody–Waite split of ln(2): ln(2) = Hi + NegLo with NegLo < 0; PosLo = -NegLo for log tails.
inline constexpr float kLn2Hi       = 0.693359375f;
inline constexpr float kLn2NegLo    = -2.12194440e-4f;
inline constexpr float kLn2PosLo    = 2.12194440e-4f;
inline constexpr float kExpArgClamp = 88.3762626647949f;
inline constexpr float kLog2e       = 1.44269504088896341f;
inline constexpr float kSqrtHalf    = 0.707106781186547524f;

// Sse_mathfun-derived SIMD helpers live in pymdp_ffi::detail (inline per TU).

namespace detail {

// One polynomial FMA Horner step: y <- c + y*x (same for exp(log·) and log(1+x)/x).
inline void __attribute__((always_inline)) simd_horner_vmlaq(float32x4_t x, float32x4_t& y, float c) noexcept {
  y = vmlaq_f32(vdupq_n_f32(c), y, x);
}

inline void __attribute__((always_inline)) simd_horner_vmlaq_x2(float32x4_t x0, float32x4_t& y0, float32x4_t x1,
                                                                float32x4_t& y1, float c) noexcept {
  const float32x4_t c_v = vdupq_n_f32(c);
  y0                    = vmlaq_f32(c_v, y0, x0);
  y1                    = vmlaq_f32(c_v, y1, x1);
}

inline void __attribute__((always_inline)) simd_exp_cw_residual(float32x4_t& x, float32x4_t fx) noexcept {
  x = vmlsq_f32(x, fx, vdupq_n_f32(kLn2Hi));
  x = vmlsq_f32(x, fx, vdupq_n_f32(kLn2NegLo));
}

inline void __attribute__((always_inline)) simd_exp_cw_residual_x2(float32x4_t& x0, float32x4_t fx0, float32x4_t& x1,
                                                                   float32x4_t fx1) noexcept {
  const float32x4_t cw1 = vdupq_n_f32(kLn2Hi);
  const float32x4_t cw2 = vdupq_n_f32(kLn2NegLo);
  x0                    = vmlsq_f32(x0, fx0, cw1);
  x1                    = vmlsq_f32(x1, fx1, cw1);
  x0                    = vmlsq_f32(x0, fx0, cw2);
  x1                    = vmlsq_f32(x1, fx1, cw2);
}

// IEEE754 reduction shared by vlog_neon_f32 / vlog_interleaved_neon_f32:
// min-normal clamp, mantissa in [0.5,1), exponent float e, sqrt(0.5) shift,
// then x <- x-1 for the log(1+x)/x polynomial. Exponent/mantissa use uint32x4
// reinterpret so shifts and masks are unsigned.
inline void __attribute__((always_inline)) simd_vlog_prepare_argument(float32x4_t& x, float32x4_t& e) noexcept {
  const float32x4_t min_norm  = vreinterpretq_f32_u32(vdupq_n_u32(kF32MinNormalBits));
  const float32x4_t sqrt_half = vdupq_n_f32(kSqrtHalf);
  const uint32x4_t  one_u     = vreinterpretq_u32_f32(vdupq_n_f32(1.0f));
  x                           = vmaxq_f32(x, min_norm);
  uint32x4_t bits             = vreinterpretq_u32_f32(x);
  uint32x4_t emm0_u           = vshrq_n_u32(bits, 23);
  uint32x4_t mant_u           = vandq_u32(bits, vdupq_n_u32(0x007fffffu));
  mant_u                      = vorrq_u32(mant_u, vdupq_n_u32(0x3f000000u));
  x                           = vreinterpretq_f32_u32(mant_u);
  int32x4_t ei                = vreinterpretq_s32_u32(emm0_u);
  ei                          = vsubq_s32(ei, vdupq_n_s32(0x7f));
  e                           = vaddq_f32(vcvtq_f32_s32(ei), vdupq_n_f32(1.0f));
  const uint32x4_t mask       = vcltq_f32(x, sqrt_half);
  x                           = vaddq_f32(x, vreinterpretq_f32_u32(vandq_u32(mask, vreinterpretq_u32_f32(x))));
  e                           = vsubq_f32(e, vreinterpretq_f32_u32(vandq_u32(mask, one_u)));
  x                           = vsubq_f32(x, vdupq_n_f32(1.0f));
}

inline void __attribute__((always_inline)) simd_vlog_prepare_argument_x2(float32x4_t& x0, float32x4_t& e0,
                                                                         float32x4_t& x1, float32x4_t& e1) noexcept {
  const float32x4_t min_norm  = vreinterpretq_f32_u32(vdupq_n_u32(kF32MinNormalBits));
  const float32x4_t sqrt_half = vdupq_n_f32(kSqrtHalf);
  const float32x4_t one_f     = vdupq_n_f32(1.0f);
  const uint32x4_t  one_u     = vreinterpretq_u32_f32(one_f);
  const uint32x4_t  mant_mask = vdupq_n_u32(0x007fffffu);
  const uint32x4_t  mant_or   = vdupq_n_u32(0x3f000000u);
  const int32x4_t   exp_bias  = vdupq_n_s32(0x7f);
  x0                          = vmaxq_f32(x0, min_norm);
  x1                          = vmaxq_f32(x1, min_norm);
  uint32x4_t bits0            = vreinterpretq_u32_f32(x0);
  uint32x4_t bits1            = vreinterpretq_u32_f32(x1);
  uint32x4_t emm0_u           = vshrq_n_u32(bits0, 23);
  uint32x4_t emm1_u           = vshrq_n_u32(bits1, 23);
  uint32x4_t mant0            = vandq_u32(bits0, mant_mask);
  uint32x4_t mant1            = vandq_u32(bits1, mant_mask);
  mant0                       = vorrq_u32(mant0, mant_or);
  mant1                       = vorrq_u32(mant1, mant_or);
  x0                          = vreinterpretq_f32_u32(mant0);
  x1                          = vreinterpretq_f32_u32(mant1);
  int32x4_t e0_i              = vreinterpretq_s32_u32(emm0_u);
  int32x4_t e1_i              = vreinterpretq_s32_u32(emm1_u);
  e0_i                        = vsubq_s32(e0_i, exp_bias);
  e1_i                        = vsubq_s32(e1_i, exp_bias);
  e0                          = vaddq_f32(vcvtq_f32_s32(e0_i), one_f);
  e1                          = vaddq_f32(vcvtq_f32_s32(e1_i), one_f);
  const uint32x4_t mask0      = vcltq_f32(x0, sqrt_half);
  const uint32x4_t mask1      = vcltq_f32(x1, sqrt_half);
  x0                          = vaddq_f32(x0, vreinterpretq_f32_u32(vandq_u32(mask0, vreinterpretq_u32_f32(x0))));
  x1                          = vaddq_f32(x1, vreinterpretq_f32_u32(vandq_u32(mask1, vreinterpretq_u32_f32(x1))));
  e0                          = vsubq_f32(e0, vreinterpretq_f32_u32(vandq_u32(mask0, one_u)));
  e1                          = vsubq_f32(e1, vreinterpretq_f32_u32(vandq_u32(mask1, one_u)));
  x0                          = vsubq_f32(x0, one_f);
  x1                          = vsubq_f32(x1, one_f);
}

// Degree-8 poly for log(1+x)/x plus Cody–Waite ln(2) recombination (x is x-1).
inline float32x4_t __attribute__((always_inline)) simd_vlog_polynomial_ln(float32x4_t x, float32x4_t e) noexcept {
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(-1.1514610310e-1f);
  simd_horner_vmlaq(x, y, 1.1676998740e-1f);
  simd_horner_vmlaq(x, y, -1.2420140846e-1f);
  simd_horner_vmlaq(x, y, 1.4249322787e-1f);
  simd_horner_vmlaq(x, y, -1.6668057665e-1f);
  simd_horner_vmlaq(x, y, 2.0000714765e-1f);
  simd_horner_vmlaq(x, y, -2.4999993993e-1f);
  simd_horner_vmlaq(x, y, 3.3333331174e-1f);
  y = vmulq_f32(y, x);
  y = vmulq_f32(y, z);
  y = vmlsq_f32(y, e, vdupq_n_f32(kLn2PosLo));
  y = vmlsq_f32(y, z, vdupq_n_f32(0.5f));
  x = vaddq_f32(x, y);
  return vmlaq_f32(x, e, vdupq_n_f32(kLn2Hi));
}

inline f32x4x2 __attribute__((always_inline)) simd_vlog_polynomial_ln_x2(float32x4_t x0, float32x4_t e0, float32x4_t x1,
                                                                         float32x4_t e1) noexcept {
  const float32x4_t ln2_pos_lo = vdupq_n_f32(kLn2PosLo);
  const float32x4_t half       = vdupq_n_f32(0.5f);
  const float32x4_t ln2_hi     = vdupq_n_f32(kLn2Hi);
  float32x4_t       z0         = vmulq_f32(x0, x0);
  float32x4_t       z1         = vmulq_f32(x1, x1);
  float32x4_t       y0         = vdupq_n_f32(-1.1514610310e-1f);
  float32x4_t       y1         = y0;
  simd_horner_vmlaq_x2(x0, y0, x1, y1, 1.1676998740e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, -1.2420140846e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, 1.4249322787e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, -1.6668057665e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, 2.0000714765e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, -2.4999993993e-1f);
  simd_horner_vmlaq_x2(x0, y0, x1, y1, 3.3333331174e-1f);
  y0 = vmulq_f32(y0, x0);
  y1 = vmulq_f32(y1, x1);
  y0 = vmulq_f32(y0, z0);
  y1 = vmulq_f32(y1, z1);
  y0 = vmlsq_f32(y0, e0, ln2_pos_lo);
  y1 = vmlsq_f32(y1, e1, ln2_pos_lo);
  y0 = vmlsq_f32(y0, z0, half);
  y1 = vmlsq_f32(y1, z1, half);
  x0 = vaddq_f32(x0, y0);
  x1 = vaddq_f32(x1, y1);
  x0 = vmlaq_f32(x0, e0, ln2_hi);
  x1 = vmlaq_f32(x1, e1, ln2_hi);
  return {x0, x1};
}

}  // namespace detail

inline float32x4_t __attribute__((always_inline)) vexp_neon_f32(float32x4_t x) noexcept {
  x = vminq_f32(x, vdupq_n_f32(kExpArgClamp));
  x = vmaxq_f32(x, vdupq_n_f32(-kExpArgClamp));
  // exp(x) = 2^k * exp(r), k = floor(x*log2(e) + 0.5), r = x - k*log(2)
  float32x4_t fx   = vmlaq_f32(vdupq_n_f32(0.5f), x, vdupq_n_f32(kLog2e));
  int32x4_t   emm0 = vcvtmq_s32_f32(fx);
  fx               = vcvtq_f32_s32(emm0);
  detail::simd_exp_cw_residual(x, fx);
  // Degree-5 polynomial for exp(r) on [-log(2)/2, log(2)/2]
  // (highest x^7 term dropped — see kVexpPolyY0 comment).
  float32x4_t z = vmulq_f32(x, x);
  float32x4_t y = vdupq_n_f32(1.3981999507e-3f);
  detail::simd_horner_vmlaq(x, y, 8.3334519073e-3f);
  detail::simd_horner_vmlaq(x, y, 4.1665795894e-2f);
  detail::simd_horner_vmlaq(x, y, 1.6666665459e-1f);
  detail::simd_horner_vmlaq(x, y, 5.0000001201e-1f);
  y = vmlaq_f32(x, y, z);
  y = vaddq_f32(y, vdupq_n_f32(1.0f));
  // 2^k via unsigned exponent bits (IEEE float32)
  uint32x4_t pow2_u = vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(emm0, vdupq_n_s32(0x7f))), 23);
  return vmulq_f32(y, vreinterpretq_f32_u32(pow2_u));
}

// Two exp evaluations with interleaved FMA steps so a shallow OoO core can
// retire independent ops from both lanes while waiting on FP latency (vs one
// long dependent chain in vexp_neon_f32).
inline f32x4x2 __attribute__((always_inline)) vexp2_neon_f32(float32x4_t x0, float32x4_t x1) noexcept {
  const float32x4_t max_x    = vdupq_n_f32(kExpArgClamp);
  const float32x4_t min_x    = vdupq_n_f32(-kExpArgClamp);
  const float32x4_t half     = vdupq_n_f32(0.5f);
  const float32x4_t log2e    = vdupq_n_f32(kLog2e);
  const float32x4_t one_f    = vdupq_n_f32(1.0f);
  const int32x4_t   exp_bias = vdupq_n_s32(0x7f);
  x0                         = vmaxq_f32(vminq_f32(x0, max_x), min_x);
  x1                         = vmaxq_f32(vminq_f32(x1, max_x), min_x);
  float32x4_t fx0            = vmlaq_f32(half, x0, log2e);
  float32x4_t fx1            = vmlaq_f32(half, x1, log2e);
  int32x4_t   k0             = vcvtmq_s32_f32(fx0);
  int32x4_t   k1             = vcvtmq_s32_f32(fx1);
  fx0                        = vcvtq_f32_s32(k0);
  fx1                        = vcvtq_f32_s32(k1);
  detail::simd_exp_cw_residual_x2(x0, fx0, x1, fx1);
  // Highest x^7 term dropped — see kVexpPolyY0 comment.
  float32x4_t z0 = vmulq_f32(x0, x0);
  float32x4_t z1 = vmulq_f32(x1, x1);
  float32x4_t y0 = vdupq_n_f32(1.3981999507e-3f);
  float32x4_t y1 = y0;
  detail::simd_horner_vmlaq_x2(x0, y0, x1, y1, 8.3334519073e-3f);
  detail::simd_horner_vmlaq_x2(x0, y0, x1, y1, 4.1665795894e-2f);
  detail::simd_horner_vmlaq_x2(x0, y0, x1, y1, 1.6666665459e-1f);
  detail::simd_horner_vmlaq_x2(x0, y0, x1, y1, 5.0000001201e-1f);
  y0                = vmlaq_f32(x0, y0, z0);
  y1                = vmlaq_f32(x1, y1, z1);
  y0                = vaddq_f32(y0, one_f);
  y1                = vaddq_f32(y1, one_f);
  uint32x4_t pow0_u = vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(k0, exp_bias)), 23);
  uint32x4_t pow1_u = vshlq_n_u32(vreinterpretq_u32_s32(vaddq_s32(k1, exp_bias)), 23);
  return {vmulq_f32(y0, vreinterpretq_f32_u32(pow0_u)), vmulq_f32(y1, vreinterpretq_f32_u32(pow1_u))};
}

inline float32x4_t __attribute__((always_inline)) vlog_neon_f32(float32x4_t x) noexcept {
  // Min-normal clamp & range reduction: detail::simd_vlog_prepare_argument.
  float32x4_t e;
  detail::simd_vlog_prepare_argument(x, e);
  return detail::simd_vlog_polynomial_ln(x, e);
}

// Natural ln for two float32x4 (same numerics as vlog_neon_f32). SIMD interleave
// lives in detail::simd_vlog_prepare_argument_x2 / simd_vlog_polynomial_ln_x2. Not log2.
inline f32x4x2 __attribute__((always_inline)) vlog_interleaved_neon_f32(float32x4_t x0, float32x4_t x1) noexcept {
  float32x4_t e0, e1;
  detail::simd_vlog_prepare_argument_x2(x0, e0, x1, e1);
  return detail::simd_vlog_polynomial_ln_x2(x0, e0, x1, e1);
}

// Specialized softmax for n == 8: mirror of softmax4 at twice the width.
// Two vld, single combined vmaxq + vmaxvq for max, vexp2 (already 8-wide)
// for exp, vaddq + vaddvq for sum, two vst. Both 4-lane vectors stay in
// register through max scan + exp + sum + normalize; exp results never
// spill. Same NaN/inf -> uniform-1/n fallback semantics as softmax_inplace.
inline void __attribute__((always_inline)) softmax8_inplace_neon(float* x) noexcept {
  float32x4_t v0 = vld1q_f32(x);
  float32x4_t v1 = vld1q_f32(x + 4);
  // NaN and +/-Inf both have all exponent bits set; OR-reduce both vectors.
  const uint32x4_t exp_mask = vdupq_n_u32(0x7f800000u);
  const uint32x4_t bad0     = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(v0), exp_mask), exp_mask);
  const uint32x4_t bad1     = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(v1), exp_mask), exp_mask);
  if (vmaxvq_u32(vorrq_u32(bad0, bad1)) != 0u) {
    fill_f32_inplace(x, 8, 0.125f);
    return;
  }
  const float       m   = vmaxvq_f32(vmaxq_f32(v0, v1));
  const float32x4_t mv  = vdupq_n_f32(m);
  f32x4x2           e   = vexp2_neon_f32(vsubq_f32(v0, mv), vsubq_f32(v1, mv));
  const float       sum = vaddvq_f32(vaddq_f32(e.a, e.b));
  if (is_nonfinite_f32(sum) || sum <= 0.0f) {
    fill_f32_inplace(x, 8, 0.125f);
    return;
  }
  const float32x4_t inv_v = vdupq_n_f32(1.0f / sum);
  vst1q_f32(x, vmulq_f32(e.a, inv_v));
  vst1q_f32(x + 4, vmulq_f32(e.b, inv_v));
}

// Specialized softmax for n == 4: single vector load/store, fused exponent-bits
// nonfinite scan, vmaxvq-based max, single vexp. Same NaN/inf -> uniform-1/n
// fallback semantics as softmax_inplace. Gated on PYMDP_FFI_VEC_LOGEXP since
// vmaxvq_* is AArch64-only and the libm path at n==4 is already cheap (the
// vector setup overhead doesn't pay off there).
inline void __attribute__((always_inline)) softmax4_inplace_neon(float* x) noexcept {
  float32x4_t v = vld1q_f32(x);
  // NaN and +/-Inf both have all exponent bits set; mantissa bits don't matter.
  const uint32x4_t exp_mask  = vdupq_n_u32(0x7f800000u);
  const uint32x4_t bad_lanes = vceqq_u32(vandq_u32(vreinterpretq_u32_f32(v), exp_mask), exp_mask);
  if (vmaxvq_u32(bad_lanes) != 0u) {
    fill_f32_inplace(x, 4, 0.25f);
    return;
  }
  const float m   = vmaxvq_f32(v);
  v               = vexp_neon_f32(vsubq_f32(v, vdupq_n_f32(m)));
  const float sum = vaddvq_f32(v);
  // After max-subtraction every lane in (-inf, 0] so exp(.) in (0, 1]; sum in
  // (0, 4] absent denormal flush. Defensive guard mirrors softmax_inplace.
  if (is_nonfinite_f32(sum) || sum <= 0.0f) {
    fill_f32_inplace(x, 4, 0.25f);
    return;
  }
  vst1q_f32(x, vmulq_f32(v, vdupq_n_f32(1.0f / sum)));
}

#endif  // PYMDP_FFI_VEC_LOGEXP

inline float entropy(const float* q, int64_t n) noexcept {
  // Contract: q must hold finite probabilities; callers typically ensure q[i] >= 0.
  // Negative or nonfinite entries are not sanitized — vector and scalar paths both
  // use raw q in the q*log(q) product (log uses max(arg, kLogEps) only).
  if (n <= 0) return 0.0f;
  float h = 0.0f;
#if PYMDP_FFI_VEC_LOGEXP
  // Interleaved vlog (two float32x4) for ILP on A57; single-vector tail.
  // 16-wide head with 4 separate FMA accumulators shortens the accumulation
  // dependency chain on production-sized n (entropy is called with num_obs
  // per modality — typically 8/16/32). 4-acc design always-on (no n>=16
  // gate): gating it back loses more of the n=16/32 win than the small
  // n=8 regression it would save.
  const float32x4_t eps_v = vdupq_n_f32(kLogEps);
  float32x4_t       acc0  = vdupq_n_f32(0.0f);
  float32x4_t       acc1  = vdupq_n_f32(0.0f);
  float32x4_t       acc2  = vdupq_n_f32(0.0f);
  float32x4_t       acc3  = vdupq_n_f32(0.0f);
  int64_t           i     = 0;
  for (; i + 16 <= n; i += 16) {
    float32x4_t q0   = vld1q_f32(q + i);
    float32x4_t q1   = vld1q_f32(q + i + 4);
    float32x4_t q2   = vld1q_f32(q + i + 8);
    float32x4_t q3   = vld1q_f32(q + i + 12);
    f32x4x2     lq01 = vlog_interleaved_neon_f32(vmaxq_f32(q0, eps_v), vmaxq_f32(q1, eps_v));
    f32x4x2     lq23 = vlog_interleaved_neon_f32(vmaxq_f32(q2, eps_v), vmaxq_f32(q3, eps_v));
    acc0             = vmlsq_f32(acc0, q0, lq01.a);
    acc1             = vmlsq_f32(acc1, q1, lq01.b);
    acc2             = vmlsq_f32(acc2, q2, lq23.a);
    acc3             = vmlsq_f32(acc3, q3, lq23.b);
  }
  for (; i + 8 <= n; i += 8) {
    float32x4_t q0 = vld1q_f32(q + i);
    float32x4_t q1 = vld1q_f32(q + i + 4);
    f32x4x2     lq = vlog_interleaved_neon_f32(vmaxq_f32(q0, eps_v), vmaxq_f32(q1, eps_v));
    acc0           = vmlsq_f32(acc0, q0, lq.a);
    acc1           = vmlsq_f32(acc1, q1, lq.b);
  }
  float32x4_t acc = vaddq_f32(vaddq_f32(acc0, acc1), vaddq_f32(acc2, acc3));
  for (; i + 4 <= n; i += 4) {
    float32x4_t qi   = vld1q_f32(q + i);
    float32x4_t qmax = vmaxq_f32(qi, eps_v);
    float32x4_t lq   = vlog_neon_f32(qmax);
    acc              = vmlsq_f32(acc, qi, lq);
  }
  h = vaddvq_f32(acc);
  for (; i < n; ++i) h -= xlogx(q[i]);
#else
  for (int64_t i = 0; i < n; ++i) h -= xlogx(q[i]);
#endif
  return h;
}

// Numerically stable in-place softmax. Per-factor state counts in pymdp
// are typically 4..32. On hosts where libm expf is fast (Apple silicon /
// any v8.2+ NEON FP16/FML host, plus modern x86), the scalar loop wins
// because the polynomial-approximation overhead exceeds the gain at
// these small n. On ARMv8.0 (Cortex-A57) libm is slow,
// so we vectorize via vexp2_neon_f32 (two-wide exp for ILP) and vexp_neon_f32
// tail. Max scan uses a 4-wide scalar unroll (four partial maxima, then combine) to shorten
// the dependency chain on in-order cores; it stays scalar with is_nonfinite_f32 + pairwise
// compares (not NEON vmaxq) so NaN/inf still force uniform fallback. Final normalize uses
// NEON vmulq + vst with a 0..3 scalar tail; the libm-only branch uses a 4-wide unrolled
// scalar multiply by inv.
inline void softmax_inplace(float* x, int64_t n) noexcept {
  if (n <= 0) return;
#if PYMDP_FFI_VEC_LOGEXP
  if (n == 8) {
    softmax8_inplace_neon(x);
    return;
  }
  if (n == 4) {
    softmax4_inplace_neon(x);
    return;
  }
#endif
  float   m0  = x[0];
  float   m1  = m0;
  float   m2  = m0;
  float   m3  = m0;
  bool    bad = is_nonfinite_f32(m0);
  int64_t i   = 1;
  for (; i + 4 <= n; i += 4) {
    const float x0 = x[i + 0];
    const float x1 = x[i + 1];
    const float x2 = x[i + 2];
    const float x3 = x[i + 3];
    bad |= is_nonfinite_f32(x0);
    bad |= is_nonfinite_f32(x1);
    bad |= is_nonfinite_f32(x2);
    bad |= is_nonfinite_f32(x3);
    m0 = x0 > m0 ? x0 : m0;
    m1 = x1 > m1 ? x1 : m1;
    m2 = x2 > m2 ? x2 : m2;
    m3 = x3 > m3 ? x3 : m3;
  }
  float m = m0;
  m       = m1 > m ? m1 : m;
  m       = m2 > m ? m2 : m;
  m       = m3 > m ? m3 : m;
  for (; i < n; ++i) {
    const float xi = x[i];
    bad |= is_nonfinite_f32(xi);
    m = xi > m ? xi : m;
  }
  if (bad) {
    fill_f32_inplace(x, n, 1.0f / static_cast<float>(n));
    return;
  }
  float sum = 0.0f;
#if PYMDP_FFI_VEC_LOGEXP
  // Production num_states = [16, 24, 32] all hit the n>=16 4-acc head; the
  // n=5/6/12 cases that can't dispatch out via softmax4/softmax8 keep the
  // 2-acc shape, so the gate stays (folding acc2/acc3 inside the gate
  // before the 8-wide tail keeps acc_sum as a single vaddq for n<16).
  const float32x4_t mv   = vdupq_n_f32(m);
  float32x4_t       acc0 = vdupq_n_f32(0.0f);
  float32x4_t       acc1 = vdupq_n_f32(0.0f);
  i                      = 0;
  if (n >= 16) {
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    for (; i + 16 <= n; i += 16) {
      float32x4_t xi0 = vld1q_f32(x + i);
      float32x4_t xi1 = vld1q_f32(x + i + 4);
      float32x4_t xi2 = vld1q_f32(x + i + 8);
      float32x4_t xi3 = vld1q_f32(x + i + 12);
      f32x4x2     e01 = vexp2_neon_f32(vsubq_f32(xi0, mv), vsubq_f32(xi1, mv));
      f32x4x2     e23 = vexp2_neon_f32(vsubq_f32(xi2, mv), vsubq_f32(xi3, mv));
      vst1q_f32(x + i, e01.a);
      vst1q_f32(x + i + 4, e01.b);
      vst1q_f32(x + i + 8, e23.a);
      vst1q_f32(x + i + 12, e23.b);
      acc0 = vaddq_f32(acc0, e01.a);
      acc1 = vaddq_f32(acc1, e01.b);
      acc2 = vaddq_f32(acc2, e23.a);
      acc3 = vaddq_f32(acc3, e23.b);
    }
    acc0 = vaddq_f32(acc0, acc2);
    acc1 = vaddq_f32(acc1, acc3);
  }
  for (; i + 8 <= n; i += 8) {
    float32x4_t xi0 = vld1q_f32(x + i);
    float32x4_t xi1 = vld1q_f32(x + i + 4);
    f32x4x2     e2  = vexp2_neon_f32(vsubq_f32(xi0, mv), vsubq_f32(xi1, mv));
    vst1q_f32(x + i, e2.a);
    vst1q_f32(x + i + 4, e2.b);
    acc0 = vaddq_f32(acc0, e2.a);
    acc1 = vaddq_f32(acc1, e2.b);
  }
  float32x4_t acc_sum = vaddq_f32(acc0, acc1);
  for (; i + 4 <= n; i += 4) {
    float32x4_t xi = vld1q_f32(x + i);
    float32x4_t e  = vexp_neon_f32(vsubq_f32(xi, mv));
    vst1q_f32(x + i, e);
    acc_sum = vaddq_f32(acc_sum, e);
  }
  sum = vaddvq_f32(acc_sum);
  for (; i < n; ++i) {
    x[i] = (expf)(x[i] - m);
    sum += x[i];
  }
#else
  for (int64_t i = 0; i < n; ++i) {
    x[i] = (expf)(x[i] - m);
    sum += x[i];
  }
#endif
  // Bit-level is_nonfinite_f32 on inputs and on sum survives -ffast-math; avoid
  // std::max / std::isfinite in the max scan so NaNs in any slot trigger uniform fallback.
  if (is_nonfinite_f32(sum) || sum <= 0.0f) {
    fill_f32_inplace(x, n, 1.0f / static_cast<float>(n));
    return;
  }
  const float inv = 1.0f / sum;
#if PYMDP_FFI_VEC_LOGEXP
  const float32x4_t inv_v = vdupq_n_f32(inv);
  int64_t           j     = 0;
  for (; j + 8 <= n; j += 8) {
    float32x4_t y0 = vmulq_f32(vld1q_f32(x + j), inv_v);
    float32x4_t y1 = vmulq_f32(vld1q_f32(x + j + 4), inv_v);
    vst1q_f32(x + j, y0);
    vst1q_f32(x + j + 4, y1);
  }
  for (; j + 4 <= n; j += 4) {
    float32x4_t y = vmulq_f32(vld1q_f32(x + j), inv_v);
    vst1q_f32(x + j, y);
  }
  for (; j < n; ++j) x[j] *= inv;
#else
  int64_t j = 0;
  for (; j + 4 <= n; j += 4) {
    x[j + 0] *= inv;
    x[j + 1] *= inv;
    x[j + 2] *= inv;
    x[j + 3] *= inv;
  }
  for (; j < n; ++j) x[j] *= inv;
#endif
}

#undef PYMDP_FFI_VEC_LOGEXP
