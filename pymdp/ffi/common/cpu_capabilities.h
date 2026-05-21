// Compile-time CPU/SIMD capability detection for pymdp FFI kernels.
//
// Preprocessor macros are kept here because they gate platform intrinsics,
// conditional includes, and build-defined features. Namespace constants expose
// the resulting capability bits to C++ callers.

#pragma once

#include <cstdint>

#if defined(__aarch64__)
#include <arm_neon.h>
#define PYMDP_FFI_HAS_AARCH64_NEON 1
#else
#define PYMDP_FFI_HAS_AARCH64_NEON 0
#endif

// Build-time override: -DPYMDP_FFI_DISABLE_F16=1 forces the f32 fallback
// even on hosts that have FP16/FML. Useful for validating the ARMv8.0
// (Cortex-A57) path on a v8.2 dev machine without cross-compiling.
#if PYMDP_FFI_HAS_AARCH64_NEON && defined(__ARM_FEATURE_FP16_FML) && !defined(PYMDP_FFI_DISABLE_F16)
#define PYMDP_FFI_HAS_F16_FML 1
#else
#define PYMDP_FFI_HAS_F16_FML 0
#endif

// x86_64 SIMD detection. AVX2 + FMA is the production modern-x86 path
// (Haswell 2013 / Zen 2018 onward).
//
// AVX-512F is opt-in via -DPYMDP_FFI_ENABLE_AVX512=1, NOT default-on even
// when the compiler defines __AVX512F__. Empirically, on x86 server CPUs
// the AVX-512 path lost to AVX2 on every pymdp fixture, including a sizable
// regression on agent_infer_policies_param_info_gain.
// Two compounding effects: (1) Intel server cores frequency-throttle on
// sustained 512-bit FMA, the throttle cost > the 2x lane-width benefit on
// the small/medium sgemv shapes here; (2) _mm512_reduce_add_ps lowers to
// a multi-step VEXTRACT + VADD chain whose cost is amortized poorly when
// the inner-n SIMD body only runs once or twice before falling to the
// scalar tail. AVX-512 may pay off on Sapphire Rapids / Zen 4 (lighter
// throttle, native horizontal-reduce ports) - opt-in lets users decide.
#if defined(__x86_64__) && defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define PYMDP_FFI_HAS_X86_AVX2 1
#else
#define PYMDP_FFI_HAS_X86_AVX2 0
#endif

#if PYMDP_FFI_HAS_X86_AVX2 && defined(__AVX512F__) && defined(PYMDP_FFI_ENABLE_AVX512)
#define PYMDP_FFI_HAS_X86_AVX512F 1
#else
#define PYMDP_FFI_HAS_X86_AVX512F 0
#endif

#define PYMDP_FFI_CAP_AARCH64_NEON 1
#define PYMDP_FFI_CAP_F16_FML 2
#define PYMDP_FFI_CAP_CUDA 4
#define PYMDP_FFI_CAP_X86_AVX2 8
#define PYMDP_FFI_CAP_X86_AVX512F 16

namespace pymdp_ffi {

// Storage type for the EFE kernel's compact A/qs path. CPUs with ARMv8.2
// FP16/FML use native f16; all other builds use float so the same kernel stays
// correct on ARMv8.0 devices.
#if PYMDP_FFI_HAS_F16_FML
using FfiF16 = __fp16;
#else
using FfiF16 = float;
#endif

inline constexpr int32_t kCpuCapabilities = (PYMDP_FFI_HAS_AARCH64_NEON ? PYMDP_FFI_CAP_AARCH64_NEON : 0) |
                                            (PYMDP_FFI_HAS_F16_FML ? PYMDP_FFI_CAP_F16_FML : 0) |
                                            (PYMDP_FFI_HAS_X86_AVX2 ? PYMDP_FFI_CAP_X86_AVX2 : 0) |
                                            (PYMDP_FFI_HAS_X86_AVX512F ? PYMDP_FFI_CAP_X86_AVX512F : 0) |
#ifdef PYMDP_FFI_HAS_CUDA
                                            PYMDP_FFI_CAP_CUDA |
#endif
                                            0;

}  // namespace pymdp_ffi
