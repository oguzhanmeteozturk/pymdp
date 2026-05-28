// Private contracts shared across the CUDA neg-EFE TUs (cache.cc / launch.cc /
// runtime.cc / entry.cc). Per-call types, small index/sig helpers, and the
// declarations for cross-TU function calls.
//
// Constraint: this header MUST NOT be transitively included from any `.cu`.
// XLA FFI's api.h (pulled via common/error_helpers.h for FfiError) is C++17-
// heavy and nvcc 10.2 on jetson rejects it ("qualified name is not allowed"
// on parametrized template member access). The CMakeLists.txt split between
// host `.cc` and device `.cu` enforces the boundary; this header lives on the
// host side.

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "common/error_helpers.h"
#include "common/kernel_primitives.h"  // fnv1a64_mix
#include "neg_efe/neg_efe_cuda_context.h"
#include "neg_efe/neg_efe_cuda_kernels.h"  // cuda_kernels::kRankMax
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"  // a_sig_bn

namespace pymdp_ffi {

// Per-call invariants threaded through run_forward and its launch helpers.
// Cache-fill and upload helpers are a separate phase and don't take this.
//
// `out_dev_ptr` is the device-side scatter destination. Host path: points
// at ctx.scratch.out_dev (owned managed buffer that gets D2H-copied to the
// host out buffer at the end). Dev path: points directly at JAX's output
// device buffer — JAX retains ownership, this code only writes through it.
// Storing JAX's pointer inside `ctx.scratch.out_dev` via CuArr::view used
// to invite ensure()/reset() to act on a non-owned buffer; threading the
// raw pointer here keeps that machinery clear of external memory entirely.
struct StageCtx {
  NegEfeContext& ctx;
  const Layout&  L;
  KernelFlags    flags;
  int            Bn;
  cudaStream_t   stream;
  float*         out_dev_ptr;
};

struct ForwardDims {
  int P;
  int T;
  int F;
  int M;
  int qsf;
};

inline ForwardDims forward_dims(const Layout& L) {
  return {static_cast<int>(L.P), static_cast<int>(L.T), static_cast<int>(L.F), static_cast<int>(L.M),
          static_cast<int>(L.qs_flat)};
}

struct QsLevel {
  std::vector<const float*> ptrs;
  std::vector<int>          histories;
};

inline size_t tm_index(int t, int m, int M) {
  return static_cast<size_t>(t) * M + m;
}

inline size_t tm_rank_dim_index(int t, int m, int i, int M) {
  return tm_index(t, m, M) * cuda_kernels::kRankMax + i;
}

// True when B-novelty (per-(t, f) factor_scores) is live for this call.
// pA-only novelty leaves wb_cache empty — skip rather than zero-and-read.
inline bool needs_factor_scores(const NegEfeContext& ctx, KernelFlags flags) {
  return flags.use_param_info_gain && !ctx.wb_cache.arrays.empty();
}

// Linear cache sig: A layout + T + Bn. C_tag captures C's content, K_m
// follows A — both are subsumed by A_sig.
inline uint64_t cuda_linear_sig(const Layout& L, int Bn) {
  return fnv1a64_mix(a_sig_bn(L, Bn), static_cast<uint64_t>(L.T));
}

// =============================================================================
// Cross-TU function declarations
// =============================================================================

// cache.cc — prepare_caches drives the per-cache fills in dependency order
// (tree, A, B, wA/wB if param_info_gain, linear). Each fill keys on the
// host-computed content_tag, so the cold path is correct regardless of
// device-pointer reuse; the warm-path content check lives in entry.cc.
FfiError prepare_caches(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, const int32_t* pm_base,
                        const float* A_base, const float* B_base, const float* C_base, const float* pA_base,
                        const float* pB_base, bool pA_present, bool pB_present);

// runtime.cc — scratch sizing, qs / inductive upload, full forward pass driver,
// optional timing log.
FfiError ensure_forward_scratch(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, cudaStream_t stream,
                                bool out_is_external);
void     upload_qs_init(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base);
FfiError upload_qs_init_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev, cudaStream_t stream);
FfiError upload_inductive_vector(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base, const float* I_base,
                                 const float* eps_base, EpsilonLayout epsilon);
FfiError upload_inductive_vector_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev,
                                     const float* I_dev, const float* eps_dev, EpsilonLayout epsilon,
                                     cudaStream_t stream);
FfiError run_forward(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, float* scatter_out_dev,
                     float* out_host_or_null, cudaStream_t stream = nullptr);
void     log_efe_cuda_timestats_if_enabled(NegEfeContext& ctx, const Layout& L, int64_t Bn);

// launch.cc — per-level launches called from run_forward.
FfiError launch_b_rollout_level(const StageCtx& stage, int t, const QsLevel& curr, QsLevel* next);
FfiError launch_modality_scores(const StageCtx& stage, int t, const QsLevel& qs_next);
FfiError launch_final_scatter(const StageCtx& stage);

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
