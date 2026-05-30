// Per-level launches for the CUDA neg-EFE forward pass: B-rollout, modality
// scoring (rank-1 thread-per-(b, h); rank-2 cuBLAS precontract + finish or
// fused tiny path; rank-3 two-stage split), and the final per-policy scatter.

#include "neg_efe/neg_efe.h"

#ifdef PYMDP_FFI_HAS_CUDA

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "xla/ffi/api/ffi.h"

#include "common/cuda_memory.h"
#include "common/error_helpers.h"
#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_cuda_context.h"
#include "neg_efe/neg_efe_cuda_internal.h"
#include "neg_efe/neg_efe_cuda_kernels.h"
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"

#define CUDA_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cuda_err(::pymdp_ffi::kEfeKernelName, op, (expr)))
#define CUBLAS_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cublas_err(::pymdp_ffi::kEfeKernelName, op, (expr)))

namespace pymdp_ffi {
namespace {

// Small-shape custom stage-1 GEMM (vs cuBLAS). Default ON: a repeatable 3-6% win
// on the CUDA-routed inductive fixtures on Orin sm_87 (agent_infer_policies
// _inductive, policy_inference_deep_inductive), parity-clean. Set
// PYMDP_FFI_SMALL_GEMM=0 to force cuBLAS (escape hatch / A-B measurement).
inline bool small_gemm_enabled() {
  static const bool on = [] {
    const char* e = std::getenv("PYMDP_FFI_SMALL_GEMM");
    return e == nullptr || e[0] != '0';  // default on; "0" disables
  }();
  return on;
}

// Gate for the warp-per-output custom kernels (stage-1 GEMM and tmp_lin), which
// win only on genuinely tiny batched GEMMs where cuBLAS's fixed ~100us 128x128
// kernel cost dominates real work. Three conditions, all required:
//   (1) small total work M*N*K — else cuBLAS's tiling amortizes;
//   (2) a small dimension min(M,N) < 32 — the case the 128x128 tile idles in;
//   (3) small K — the warp does a single stride-32 reduction over K with an
//       uncoalesced stride-N read of the other operand, so large K loses to
//       cuBLAS's K-tiling. Measured: K=4096 (mmp_many_modalities_highdim, rank-2
//       S0*S1=64*64) regressed forced-CUDA +24% vs cuBLAS; K~768 (rollout /
//       inductive) wins. Cap sits between.
constexpr int     kSmallGemmMaxMinDim = 32;
constexpr int64_t kSmallGemmMaxWork   = 1 << 20;  // 1M MACs
constexpr int     kSmallGemmMaxK      = 1024;

inline bool use_warp_gemm(int M, int N, int K) {
  if (!small_gemm_enabled()) return false;
  if (K > kSmallGemmMaxK) return false;
  const int64_t work = static_cast<int64_t>(M) * N * K;
  return work <= kSmallGemmMaxWork && (M < kSmallGemmMaxMinDim || N < kSmallGemmMaxMinDim);
}

// Computes row-major out[b, m, n] = a_rm[b, m, k] @ q01_outer[b, k, n].
// The cuBLAS call is column-major, so the GEMM is expressed as C^T = Q^T @ A^T.
// Small shapes route to a warp-per-output custom kernel (see use_small_q01_gemm).
FfiError run_batched_q01_gemm(const StageCtx& stage, const char* op, const float* q01_outer, const float* a_rm,
                              float* out_rm, int M_rm, int N_rm, int K_rm) {
  if (use_warp_gemm(M_rm, N_rm, K_rm)) {
    CUDA_TRY(op, cuda_kernels::launch_small_batched_gemm_rm(a_rm, q01_outer, out_rm, stage.Bn, M_rm, N_rm, K_rm,
                                                            stage.stream));
    return FfiError::Success();
  }
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  CUBLAS_TRY(op, cublasSgemmStridedBatched(stage.ctx.cublas_handle.handle, CUBLAS_OP_N, CUBLAS_OP_N, N_rm, M_rm, K_rm,
                                           &alpha, q01_outer, N_rm, static_cast<long long>(K_rm) * N_rm, a_rm, K_rm,
                                           static_cast<long long>(M_rm) * K_rm, &beta, out_rm, N_rm,
                                           static_cast<long long>(M_rm) * N_rm, stage.Bn));
  return FfiError::Success();
}

// Row-major tmp_lin[b, h, s] = sum_k q01_outer[b, k, h] * linear[b, k, s].
// cuBLAS reads q01_outer as col-major [H_kk, K_keep] under OP_T, linear as
// col-major [S_split, K_keep] under OP_N; the col-major [S_split, H_kk]
// output is bytewise row-major [H_kk, S_split]. Caller gates on H_kk —
// dispatch beats the per-(b, h, s) serial-K kernel at H_kk >= ~32 on sm_53.
FfiError run_batched_tmp_lin_gemm(const StageCtx& stage, const float* q01_outer, const float* linear, float* tmp_lin,
                                  int K_keep, int H_kk, int S_split) {
  const float alpha = 1.0f;
  const float beta  = 0.0f;
  CUBLAS_TRY("rank-3 tmp_lin GEMM",
             cublasSgemmStridedBatched(stage.ctx.cublas_handle.handle, CUBLAS_OP_N, CUBLAS_OP_T, S_split, H_kk, K_keep,
                                       &alpha, linear, S_split, static_cast<long long>(K_keep) * S_split, q01_outer,
                                       H_kk, static_cast<long long>(K_keep) * H_kk, &beta, tmp_lin, S_split,
                                       static_cast<long long>(H_kk) * S_split, stage.Bn));
  return FfiError::Success();
}

// All pointers carry __restrict__ — each field aliases a distinct cache /
// scratch slot, so the no-alias guarantee can propagate into inner loops.
// H[i] for i in [deps.rank, kRankMax) is 1-padded by build_modality_score
// _layout; rank-{1,2} kernels ignore the trailing entries.
struct ModalityLaunch {
  DependencyView deps;
  const float* __restrict__ A_unflat;
  const float* __restrict__ wA_unflat;
  const float* __restrict__ wA_cublas;
  const float* __restrict__ linear;
  float* __restrict__ score_out;
  int                                     O;
  std::array<int, cuda_kernels::kRankMax> H;
  bool                                    use_states;
  bool                                    use_linear;
  bool                                    use_pA;
};

ModalityLaunch make_modality_launch(const StageCtx& stage, int t, int m) {
  NegEfeContext&     ctx        = stage.ctx;
  const Layout&      L          = stage.L;
  const KernelFlags& flags      = stage.flags;
  const int          M          = static_cast<int>(L.M);
  const FfiInt64Vec& mod_off    = ctx.tree_cache.mod_score_offsets;
  const FfiInt32Vec& mod_h_dims = ctx.tree_cache.mod_h_dims;
  ModalityLaunch     ml{};
  ml.deps     = modality_state_deps(L, m);
  ml.A_unflat = ctx.a_cache.arrays[m].as<const float>();
  // wA cache may be shorter than M (e.g. only the modalities that actually
  // have pA filled) — `!empty()` is not sufficient. Index by m only after
  // bounds-checking, otherwise this is a host-side OOB read that can SEGV.
  ml.wA_unflat  = (flags.use_param_info_gain && static_cast<size_t>(m) < ctx.wa_cache.arrays.size())
                      ? ctx.wa_cache.arrays[m].as<const float>()
                      : nullptr;
  ml.wA_cublas  = (flags.use_param_info_gain && static_cast<size_t>(m) < ctx.wa_cache.cublas_views.size() &&
                   ctx.wa_cache.cublas_views[m].ptr != nullptr)
                      ? ctx.wa_cache.cublas_views[m].as<const float>()
                      : nullptr;
  ml.score_out  = ctx.scratch.scores_concat.as<float>() + mod_off[tm_index(t, m, M)];
  ml.O          = static_cast<int>(L.O[m]);
  ml.use_states = flags.use_states_info_gain;
  ml.use_linear = flags.use_states_info_gain || flags.use_utility;
  ml.use_pA     = ml.wA_unflat != nullptr;
  ml.linear     = ml.use_linear ? ctx.linear_cache.per_tm[t][m].as<const float>() : nullptr;
  for (int i = 0; i < cuda_kernels::kRankMax; ++i) {
    ml.H[i] = mod_h_dims[tm_rank_dim_index(t, m, i, M)];
  }
  return ml;
}

FfiError build_q01_outer_for_modality(const StageCtx& stage, const QsLevel& qs_next, const ModalityLaunch& ml,
                                      float* q01_outer) {
  const int Sd0 = dep_state_size(stage.L, ml.deps, 0);
  const int Sd1 = dep_state_size(stage.L, ml.deps, 1);
  CUDA_TRY("build_qs01_outer",
           cuda_kernels::launch_build_qs01_outer(qs_next.ptrs[ml.deps.factors[0]], qs_next.ptrs[ml.deps.factors[1]],
                                                 stage.Bn, ml.H[0], ml.H[1], Sd0, Sd1, q01_outer, stage.stream));
  return FfiError::Success();
}

constexpr int64_t kTinyFusedModalityMaxWork = 262144;
constexpr int     kTinyFusedModalityMaxK    = 64;
constexpr int     kTinyFusedModalityMaxO    = 16;

inline bool use_tiny_fused_modality_path(const ModalityLaunch& ml, int Bn, int H_total, int K) {
  // K-loop passes per output history: qo (states or pA) + wA (pA) + linear.
  // When all terms are off the fused kernel still wins (one cheap zero-write,
  // no q01_outer materialization).
  int passes = 0;
  if (ml.use_states || ml.use_pA) passes += ml.O;
  if (ml.use_pA) passes += ml.O;
  if (ml.use_linear) passes += 1;
  if (passes == 0) return true;

  const int64_t work = static_cast<int64_t>(Bn) * H_total * K * passes;
  return K <= kTinyFusedModalityMaxK && ml.O <= kTinyFusedModalityMaxO && work <= kTinyFusedModalityMaxWork;
}

FfiError launch_modality_rank1(const StageCtx& stage, const QsLevel& qs_next, const ModalityLaunch& ml,
                               int64_t total_mod_entries) {
  const int Sd0 = dep_state_size(stage.L, ml.deps, 0);
  CUDA_TRY("modality_score_dedup_rank1",
           cuda_kernels::launch_modality_score_dedup_rank1(
               ml.A_unflat, ml.wA_unflat, ml.linear, qs_next.ptrs[ml.deps.factors[0]], stage.Bn, ml.O, ml.H[0], Sd0,
               total_mod_entries, ml.use_states, ml.use_linear, ml.use_pA, ml.score_out, stage.stream));
  return FfiError::Success();
}

// lin_override (when non-null) supplies this modality's linear term already
// contracted into [.., H_kk] with per-batch stride lin_b_stride — used when a
// dependency group's linear terms were batched into one stacked GEMM upstream
// (see launch_rank2_group_linear). When null, the per-modality linear GEMM runs
// here with stride H_kk.
FfiError launch_modality_rank2(const StageCtx& stage, const QsLevel& qs_next, const ModalityLaunch& ml,
                               int64_t total_mod_entries, bool build_q01, const float* lin_override = nullptr,
                               int lin_b_stride = 0) {
  NegEfeContext& ctx  = stage.ctx;
  const int      Sd0  = dep_state_size(stage.L, ml.deps, 0);
  const int      Sd1  = dep_state_size(stage.L, ml.deps, 1);
  const int      K_d  = Sd0 * Sd1;
  const int      H_kk = ml.H[0] * ml.H[1];

  if (use_tiny_fused_modality_path(ml, stage.Bn, H_kk, K_d)) {
    CUDA_TRY("modality_score_dedup_rank2_fused_tiny",
             cuda_kernels::launch_modality_score_dedup_rank2_fused_tiny(
                 ml.A_unflat, ml.wA_unflat, ml.linear, qs_next.ptrs[ml.deps.factors[0]],
                 qs_next.ptrs[ml.deps.factors[1]], stage.Bn, ml.O, ml.H[0], ml.H[1], Sd0, Sd1, total_mod_entries,
                 ml.use_states, ml.use_linear, ml.use_pA, ml.score_out, stage.stream));
    return FfiError::Success();
  }

  float* q01_outer = ctx.scratch.q01_outer.as<float>();
  float* tmp_qo_cb = ctx.scratch.tmp_qo_cublas.as<float>();
  float* tmp_wa_cb = ml.use_pA ? ctx.scratch.tmp_wa_cublas.as<float>() : nullptr;

  // q01_outer = qs0⊗qs1 depends only on the dependency group (factors + H + S),
  // not on the modality, so the caller skips the rebuild for a run of modalities
  // that share a group (build_q01=false reuses the resident buffer).
  if (build_q01) PYMDP_TRY(build_q01_outer_for_modality(stage, qs_next, ml, q01_outer));
  if (ml.use_states || ml.use_pA) {
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 modality GEMM", q01_outer, ml.A_unflat, tmp_qo_cb, ml.O, H_kk, K_d));
  }
  if (ml.use_pA) {
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 pA GEMM", q01_outer, ml.wA_unflat, tmp_wa_cb, ml.O, H_kk, K_d));
  }

  const float* tmp_lin = lin_override;
  int          lin_str = lin_b_stride;
  if (ml.use_linear && lin_override == nullptr) {
    // Per-modality linear term: GEMM (linear[b,1,K] @ q01_outer[b,K,H_kk] ->
    // tmp_lin[b,1,H_kk]) so the K_d reduction runs on cuBLAS rather than the
    // finish kernel's Bn*H_kk-thread serial loop. A group of >=2 modalities
    // batches this into one GEMM upstream and passes lin_override instead.
    float* per_mod_lin = ctx.scratch.split_tmp_lin.as<float>();
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 linear GEMM", q01_outer, ml.linear, per_mod_lin, 1, H_kk, K_d));
    tmp_lin = per_mod_lin;
    lin_str = H_kk;
  }
  CUDA_TRY("modality_score_dedup_rank2_cublas_finish",
           cuda_kernels::launch_modality_score_dedup_rank2_cublas_finish(
               tmp_qo_cb, tmp_wa_cb, tmp_lin, stage.Bn, ml.O, H_kk, lin_str, total_mod_entries, ml.use_states,
               ml.use_linear, ml.use_pA, ml.score_out, stage.stream));
  return FfiError::Success();
}

FfiError launch_modality_rank3(const StageCtx& stage, const QsLevel& qs_next, const ModalityLaunch& ml, int m,
                               int64_t total_mod_entries) {
  NegEfeContext& ctx      = stage.ctx;
  const int      Sd0      = dep_state_size(stage.L, ml.deps, 0);
  const int      Sd1      = dep_state_size(stage.L, ml.deps, 1);
  const int      S_split  = dep_state_size(stage.L, ml.deps, 2);
  const int      K_keep   = Sd0 * Sd1;
  const int      H_keep_0 = ml.H[0];
  const int      H_keep_1 = ml.H[1];
  const int      H_split  = ml.H[2];
  const int      H_kk     = H_keep_0 * H_keep_1;
  const int      H_full   = H_kk * H_split;

  if (use_tiny_fused_modality_path(ml, stage.Bn, H_full, K_keep * S_split)) {
    CUDA_TRY("modality_score_dedup_rank3_fused_tiny",
             cuda_kernels::launch_modality_score_dedup_rank3_fused_tiny(
                 ml.A_unflat, ml.wA_unflat, ml.linear, qs_next.ptrs[ml.deps.factors[0]],
                 qs_next.ptrs[ml.deps.factors[1]], qs_next.ptrs[ml.deps.factors[2]], stage.Bn, ml.O, H_keep_0, H_keep_1,
                 H_split, Sd0, Sd1, S_split, total_mod_entries, ml.use_states, ml.use_linear, ml.use_pA, ml.score_out,
                 stage.stream));
    return FfiError::Success();
  }

  if (m < 0 || static_cast<size_t>(m) >= ctx.a_cache.cublas_views.size() ||
      ctx.a_cache.cublas_views[static_cast<size_t>(m)].ptr == nullptr) {
    return invalid_arg(kEfeKernelName, "rank-3 modality requires A cuBLAS layout (internal error)");
  }
  if (ml.use_pA && (static_cast<size_t>(m) >= ctx.wa_cache.cublas_views.size() ||
                    ctx.wa_cache.cublas_views[static_cast<size_t>(m)].ptr == nullptr)) {
    return invalid_arg(kEfeKernelName, "rank-3 pA requires wA cuBLAS layout (internal error)");
  }

  const float* A_cb        = ctx.a_cache.cublas_views[static_cast<size_t>(m)].as<const float>();
  const float* wA_cb       = ml.use_pA ? ctx.wa_cache.cublas_views[static_cast<size_t>(m)].as<const float>() : nullptr;
  float*       q01_outer   = ctx.scratch.q01_outer.as<float>();
  float*       tmp_qo_cb   = ctx.scratch.tmp_qo_cublas.as<float>();
  float*       split_qo    = ctx.scratch.split_tmp_qo.as<float>();
  float*       tmp_lin_buf = ctx.scratch.split_tmp_lin.as<float>();

  PYMDP_TRY(build_q01_outer_for_modality(stage, qs_next, ml, q01_outer));

  if (ml.use_states || ml.use_pA) {
    PYMDP_TRY(
        run_batched_q01_gemm(stage, "rank-3 modality GEMM", q01_outer, A_cb, tmp_qo_cb, ml.O * S_split, H_kk, K_keep));
    CUDA_TRY("tmp_qo_cublas_to_my", cuda_kernels::launch_tmp_qo_cublas_to_my(tmp_qo_cb, stage.Bn, ml.O, S_split, H_kk,
                                                                             split_qo, stage.stream));
  }

  const float* tmp_wa_split = nullptr;
  if (ml.use_pA) {
    float* tmp_wa_cb = ctx.scratch.tmp_wa_cublas.as<float>();
    float* split_wa  = ctx.scratch.split_tmp_wa.as<float>();
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-3 pA GEMM", q01_outer, wA_cb, tmp_wa_cb, ml.O * S_split, H_kk, K_keep));
    CUDA_TRY("tmp_wa_cublas_to_my", cuda_kernels::launch_tmp_qo_cublas_to_my(tmp_wa_cb, stage.Bn, ml.O, S_split, H_kk,
                                                                             split_wa, stage.stream));
    tmp_wa_split = split_wa;
  }

  if (ml.use_linear) {
    // tmp_lin is a batched GEMM out[h, s2] = sum_k q01[k, h] * linear[k, s2]
    // (M = H_kk, N = S_split, K = K_keep). Tiny shapes take the warp-per-output
    // kernel (same lever as the stage-1 GEMM); above H_kk=32 cuBLAS's tiling
    // pays off; the thread-per-output per_h kernel covers the rest.
    constexpr int kTmpLinCublasMinHkk = 32;
    if (use_warp_gemm(H_kk, S_split, K_keep)) {
      CUDA_TRY("tmp_lin_warp", cuda_kernels::launch_tmp_lin_warp(q01_outer, ml.linear, stage.Bn, K_keep, H_kk, S_split,
                                                                 tmp_lin_buf, stage.stream));
    } else if (H_kk >= kTmpLinCublasMinHkk) {
      PYMDP_TRY(run_batched_tmp_lin_gemm(stage, q01_outer, ml.linear, tmp_lin_buf, K_keep, H_kk, S_split));
    } else {
      CUDA_TRY("tmp_lin_per_h", cuda_kernels::launch_tmp_lin_per_h(q01_outer, ml.linear, stage.Bn, K_keep, H_kk,
                                                                   S_split, tmp_lin_buf, stage.stream));
    }
  }

  CUDA_TRY("modality_score_dedup_rank3_stage2",
           cuda_kernels::launch_modality_score_dedup_rank3_stage2(
               (ml.use_states || ml.use_pA) ? split_qo : nullptr, ml.use_linear ? tmp_lin_buf : nullptr, tmp_wa_split,
               qs_next.ptrs[ml.deps.factors[2]], stage.Bn, ml.O, H_keep_0, H_keep_1, H_split, S_split,
               total_mod_entries, ml.use_states, ml.use_linear, ml.use_pA, ml.score_out, stage.stream));
  return FfiError::Success();
}

}  // namespace

FfiError launch_b_rollout_level(const StageCtx& stage, int t, const QsLevel& curr, QsLevel* next) {
  NegEfeContext&             ctx                       = stage.ctx;
  const Layout&              L                         = stage.L;
  const int                  F                         = static_cast<int>(L.F);
  const int                  write_slot                = t & 1;
  const FactorHistoryLevels& ftree                     = ctx.tree_cache.factor_tree;
  const CuArrGrid2D&         factor_parent_history     = ctx.tree_cache.factor_parent_history;
  const CuArrGrid2D&         factor_action_per_history = ctx.tree_cache.factor_action_per_history;

  next->ptrs.resize(F);
  next->histories.resize(F);
  // Factor-score gate is f-invariant — resolve once outside the loop.
  const bool do_fs = needs_factor_scores(ctx, stage.flags);
  for (int f = 0; f < F; ++f) {
    const int            Sf       = static_cast<int>(L.S[f]);
    const int            Uf       = static_cast<int>(L.U[f]);
    const int            Hf_t     = ftree[t][f].n_histories;
    const DependencyView deps     = factor_transition_deps(L, f);
    const int            n_par    = deps.rank;
    const int            K_f      = static_cast<int>(b_K(L, f));
    const float*         B_f      = ctx.b_cache.arrays[f].as<const float>();
    const int32_t*       action_h = factor_action_per_history[t][f].as<const int32_t>();
    const int32_t*       parent_h = factor_parent_history[t][f].as<const int32_t>();
    float*               qs_out   = ctx.scratch.qs_factor_buf[write_slot][f].as<float>();
    float*               factor_score =
        do_fs
            ? (ctx.scratch.factor_scores.as<float>() + ctx.tree_cache.ind_score_offsets[static_cast<size_t>(t) * F + f])
            : nullptr;
    const float* wB_f = do_fs ? ctx.wb_cache.arrays[f].as<const float>() : nullptr;

    // Inductive is folded into the B-rollout phase-2 reduction when
    // use_inductive is on; nullptr disables the inner reduction in the kernel.
    const float*  v_full        = stage.flags.use_inductive ? ctx.scratch.v_full_dev.as<const float>() : nullptr;
    float*        ind_score_t_f = stage.flags.use_inductive
                                      ? (ctx.scratch.inductive_concat.as<float>() +
                                         ctx.tree_cache.ind_score_offsets[static_cast<size_t>(t) * F + f])
                                      : nullptr;
    const int     qs_flat       = static_cast<int>(L.qs_flat);
    const int     qs_off_f      = static_cast<int>(L.qs_off[f]);
    const int64_t ind_b_stride  = ctx.tree_cache.total_ind_entries;

    cuda_kernels::BRolloutParents parents{};
    for (int i = 0; i < n_par; ++i) {
      const int64_t pf = deps.factors[i];
      parents.qs[i]    = curr.ptrs[pf];
      parents.H[i]     = curr.histories[pf];
      parents.S[i]     = static_cast<int>(L.S[pf]);
    }
    CUDA_TRY("b_rollout_general",
             cuda_kernels::launch_b_rollout_general(B_f, wB_f, v_full, qs_flat, qs_off_f, action_h, parent_h, parents,
                                                    n_par, stage.Bn, Hf_t, Sf, K_f, Uf, qs_out, factor_score,
                                                    ind_score_t_f, ind_b_stride, stage.stream));
    next->ptrs[f]      = qs_out;
    next->histories[f] = Hf_t;
  }
  return FfiError::Success();
}

// Signature identifying a rank-2 q01_outer (= qs0⊗qs1): the two dependency
// factors and their per-level history/state dims. Two rank-2 modalities with
// equal signatures at the same level share an identical q01_outer.
struct Q01Sig {
  int64_t f0 = -1, f1 = -1;
  int     H0 = -1, H1 = -1, S0 = -1, S1 = -1;
  bool    valid = false;
  bool    operator==(const Q01Sig& o) const {
    return valid && o.valid && f0 == o.f0 && f1 == o.f1 && H0 == o.H0 && H1 == o.H1 && S0 == o.S0 && S1 == o.S1;
  }
};

// Lever #2: gather a run of g consecutive rank-2 modalities sharing one
// dependency group into stacked_lin_in[Bn, g, K_d] (a cudaMemcpy2DAsync per
// member handles the b/g/k interleave), then one cuBLAS GEMM against the shared
// q01_outer writes stacked_lin_out[Bn, g, H_kk]. Replaces g skinny (M_rm=1, big
// K) per-modality GEMMs — a pathological cuBLAS shape — with one M_rm=g GEMM.
// q01_outer must already hold this group's signature.
static FfiError launch_rank2_group_linear(const StageCtx& stage, const std::vector<ModalityLaunch>& mls, int m0, int g,
                                          int K_d, int H_kk) {
  NegEfeContext& ctx         = stage.ctx;
  float*         stacked_in  = ctx.scratch.stacked_lin_in.as<float>();
  float*         stacked_out = ctx.scratch.stacked_lin_out.as<float>();
  const size_t   dpitch      = static_cast<size_t>(g) * K_d * sizeof(float);
  const size_t   rowbytes    = static_cast<size_t>(K_d) * sizeof(float);
  for (int j = 0; j < g; ++j) {
    CUDA_TRY("stacked linear gather",
             cudaMemcpy2DAsync(stacked_in + static_cast<size_t>(j) * K_d, dpitch, mls[m0 + j].linear, rowbytes,
                               rowbytes, stage.Bn, cudaMemcpyDeviceToDevice, stage.stream));
  }
  PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 group linear GEMM", ctx.scratch.q01_outer.as<float>(), stacked_in,
                                 stacked_out, g, H_kk, K_d));
  return FfiError::Success();
}

FfiError launch_modality_scores(const StageCtx& stage, int t, const QsLevel& qs_next) {
  const int     M                 = static_cast<int>(stage.L.M);
  const int64_t total_mod_entries = stage.ctx.tree_cache.total_mod_entries;

  // Per-modality launches + rank-2 dependency-group signatures (for grouping)
  // and tiny-path verdicts (tiny depends on O, which can vary within a group).
  std::vector<ModalityLaunch> mls;
  mls.reserve(M);
  std::vector<Q01Sig> sigs(M);
  std::vector<char>   tiny(M, 0);
  for (int m = 0; m < M; ++m) {
    mls.push_back(make_modality_launch(stage, t, m));
    const ModalityLaunch& ml = mls[m];
    if (ml.deps.rank == 2) {
      sigs[m] = Q01Sig{ml.deps.factors[0],
                       ml.deps.factors[1],
                       ml.H[0],
                       ml.H[1],
                       dep_state_size(stage.L, ml.deps, 0),
                       dep_state_size(stage.L, ml.deps, 1),
                       true};
      tiny[m] = use_tiny_fused_modality_path(ml, stage.Bn, sigs[m].H0 * sigs[m].H1, sigs[m].S0 * sigs[m].S1) ? 1 : 0;
    }
  }

  Q01Sig last{};  // signature currently resident in q01_outer (lever #1 memo)
  int    m = 0;
  while (m < M) {
    const ModalityLaunch& ml = mls[m];

    // Lever #2: batch the linear term across a run of same-group non-tiny
    // rank-2 modalities into one GEMM. Only when use_linear (the term exists).
    if (ml.deps.rank == 2 && !tiny[m] && ml.use_linear) {
      int g = 1;
      while (m + g < M && mls[m + g].deps.rank == 2 && !tiny[m + g] && (sigs[m + g] == sigs[m])) ++g;
      if (g >= 2) {
        const int K_d  = sigs[m].S0 * sigs[m].S1;
        const int H_kk = sigs[m].H0 * sigs[m].H1;
        // Lever #1: build the shared q01_outer once for the whole group.
        PYMDP_TRY(build_q01_outer_for_modality(stage, qs_next, ml, stage.ctx.scratch.q01_outer.as<float>()));
        // Lever #2: batch the linear term into one GEMM across the group. (The
        // A-term GEMM stays per-modality: batching it loses — its output is
        // N=H_kk=16, so a stacked GEMM only wastes wider tiles, and re-gathering
        // the large A every call costs more than it saves.)
        PYMDP_TRY(launch_rank2_group_linear(stage, mls, m, g, K_d, H_kk));
        const float* lin_out = stage.ctx.scratch.stacked_lin_out.as<float>();
        for (int j = 0; j < g; ++j) {
          PYMDP_TRY(launch_modality_rank2(stage, qs_next, mls[m + j], total_mod_entries, /*build_q01=*/false,
                                          /*lin_override=*/lin_out + static_cast<size_t>(j) * H_kk,
                                          /*lin_b_stride=*/g * H_kk));
        }
        last = sigs[m];  // q01_outer now holds this group's signature
        m += g;
        continue;
      }
    }

    switch (ml.deps.rank) {
    case 1:
      PYMDP_TRY(launch_modality_rank1(stage, qs_next, ml, total_mod_entries));
      last = Q01Sig{};
      break;
    case 2: {
      // Lever #1 single-path memo: reuse q01_outer when this rank-2 modality
      // shares the resident signature (covers the use_linear==false case that
      // the group path above skips).
      const bool reuse = !tiny[m] && (sigs[m] == last);
      PYMDP_TRY(launch_modality_rank2(stage, qs_next, ml, total_mod_entries, /*build_q01=*/!reuse));
      last = tiny[m] ? Q01Sig{} : sigs[m];
      break;
    }
    case 3:
      PYMDP_TRY(launch_modality_rank3(stage, qs_next, ml, m, total_mod_entries));
      last = Q01Sig{};
      break;
    default:
      return invalid_arg(kEfeKernelName, "CUDA path supports modality rank in [1, 3]");
    }
    ++m;
  }
  return FfiError::Success();
}

FfiError launch_final_scatter(const StageCtx& stage) {
  NegEfeContext&     ctx   = stage.ctx;
  const KernelFlags& flags = stage.flags;
  const ForwardDims  d     = forward_dims(stage.L);
  const bool         do_fs = needs_factor_scores(ctx, flags);
  CUDA_TRY("final_scatter_dedup",
           cuda_kernels::launch_final_scatter_dedup(
               ctx.scratch.scores_concat.as<const float>(),
               flags.use_inductive ? ctx.scratch.inductive_concat.as<const float>() : nullptr,
               do_fs ? ctx.scratch.factor_scores.as<const float>() : nullptr,
               ctx.tree_cache.policy_to_modality_idx_dev.as<const int32_t>(),
               ctx.tree_cache.factor_policy_to_history_dev.as<const int32_t>(),
               ctx.tree_cache.mod_score_offsets_dev.as<const int64_t>(),
               (flags.use_inductive || do_fs) ? ctx.tree_cache.ind_score_offsets_dev.as<const int64_t>() : nullptr,
               stage.Bn, d.T, d.M, d.F, d.P, ctx.tree_cache.total_mod_entries, ctx.tree_cache.total_ind_entries,
               flags.use_inductive, do_fs, stage.out_dev_ptr, stage.stream));
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
