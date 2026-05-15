// CUDA neg-EFE kernel. Two XLA registrations share the pipeline below; the
// selection is made from Python via PYMDP_FFI_USE_CUDA + presence of a CUDA
// JAX backend:
//   * NegEfeCudaHost (platform="cpu")  — host buffers in/out, used wherever
//     JAX runs on CPU. D2H copy at the end.
//   * NegEfeCudaDev  (platform="CUDA") — JAX device buffers in/out, scatter
//     writes straight into JAX's output, no D2H.
//
// Forward pipeline (run_forward):
//   1. per-(t, f) B-rollout from cached factor histories
//   2. per-(t, m) modality scoring — rank-1 thread-per-(b, h), rank-2 cuBLAS
//      precontract + finish, rank-3 two-stage split (precontract two factors,
//      finish + entropy with the third)
//   3. per-(t, f) inductive contribution (folded into B-rollout phase 2)
//   4. final scatter: per (b, p) sum of (t, m) modality scores + (t, f)
//      inductive scores via per-policy history-tuple lookups
//
// All device-visible storage is cudaMallocManaged; on Tegra that maps to
// shared DRAM so host packing + a single device sync covers the
// single-stream pipeline.

#include "neg_efe.h"

#ifdef PYMDP_FFI_HAS_CUDA

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "xla/ffi/api/ffi.h"

#include "cuda_host_alias.h"
#include "factor_history_tables.h"
#include "factor_history_tree.h"
#include "neg_efe_cuda_context.h"
#include "neg_efe_cuda_kernels.h"
#include "neg_efe_cuda_memory.h"
#include "neg_efe_entry.h"
#include "neg_efe_layout.h"
#include "neg_efe_precompute.h"
#include "error_helpers.h"
#include "kernel_primitives.h"

namespace ffi = ::xla::ffi;

namespace pymdp_ffi {
namespace {

// Factor-history dedup model
// --------------------------
// Each factor's predicted qs is keyed by (parent histories, action). Scoring
// and inductive paths index by these per-factor histories; the final scatter
// maps per-history scores back to policies via lookup tables built once per
// pm signature. At t=0 there is no prior level — parent_h is 0 for all
// parents, qs is pulled from the shared qs_init buffer (H=1 implicit).

// kRankMax: per-modality dependency-rank cap (mirrors neg_efe_cuda_kernels.h).
static_assert(cuda_kernels::kRankMax == 3, "kRankMax assumed = 3 by index encoding");

inline FfiError ensure_cublas_handle(NegEfeContext& ctx) {
  if (ctx.cublas_handle.handle != nullptr) return FfiError::Success();
  CUBLAS_TRY("cublasCreate", cublasCreate(&ctx.cublas_handle.handle));
  return FfiError::Success();
}

inline size_t tm_index(int t, int m, int M) {
  return static_cast<size_t>(t) * M + m;
}

// True when B-novelty (per-(t, f) factor_scores) is live for this call.
// pA-only novelty leaves wb_cache empty — skip rather than zero-and-read.
inline bool needs_factor_scores(const NegEfeContext& ctx, KernelFlags flags) {
  return flags.use_param_info_gain && !ctx.wb_cache.arrays.empty();
}

inline size_t tm_rank_dim_index(int t, int m, int i, int M) {
  return tm_index(t, m, M) * cuda_kernels::kRankMax + i;
}

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

inline CuArr append_packed_slices(CuPool* pool, std::vector<float>* scratch, const float* base, int Bn,
                                  int64_t batch_stride, int64_t slice_offset, size_t slice_size) {
  pack_batched_slices(scratch, base, Bn, batch_stride, slice_offset, slice_size);
  return pool->append_copy(scratch->data(), scratch->size() * sizeof(float));
}

inline void pack_rank3_cublas_view(const float* packed, int Bn, int O_m, int K_keep, int S_split, size_t per_batch,
                                   std::vector<float>* cublas_packed) {
  cublas_packed->assign(static_cast<size_t>(Bn) * O_m * S_split * K_keep, 0.0f);
  for (int b = 0; b < Bn; ++b) {
    const float* src_b = packed + static_cast<size_t>(b) * per_batch;
    float*       dst_b = cublas_packed->data() + static_cast<size_t>(b) * O_m * S_split * K_keep;
    for (int o = 0; o < O_m; ++o) {
      for (int k = 0; k < K_keep; ++k) {
        for (int s = 0; s < S_split; ++s) {
          dst_b[(o * S_split + s) * K_keep + k] = src_b[(o * K_keep + k) * S_split + s];
        }
      }
    }
  }
}

inline CuArr append_rank3_cublas_view(CuPool* pool, const std::vector<float>& packed, int Bn, int O_m, int K_keep,
                                      int S_split, size_t per_batch, std::vector<float>* cublas_packed) {
  pack_rank3_cublas_view(packed.data(), Bn, O_m, K_keep, S_split, per_batch, cublas_packed);
  return pool->append_copy(cublas_packed->data(), cublas_packed->size() * sizeof(float));
}

// Bytes needed in a CuPool for an A-shaped pack: per-modality A (always) +
// rank-3 cuBLAS-permuted view (same size as A, rank-3 modalities only).
inline size_t a_pack_pool_bytes(const Layout& L, int Bn) {
  size_t total = 0;
  for (int m = 0; m < static_cast<int>(L.M); ++m) {
    const size_t per_batch_bytes = static_cast<size_t>(Bn) * a_size(L, m) * sizeof(float);
    total += round_up_8(per_batch_bytes);
    if (modality_state_deps(L, m).rank == 3) total += round_up_8(per_batch_bytes);
  }
  return total;
}

// Bytes needed in a CuPool for a B-shaped pack: per-factor B.
inline size_t b_pack_pool_bytes(const Layout& L, int Bn) {
  size_t total = 0;
  for (int f = 0; f < static_cast<int>(L.F); ++f) {
    total += round_up_8(static_cast<size_t>(Bn) * b_size(L, f) * sizeof(float));
  }
  return total;
}

// Pack per-modality A-shaped data from `src` (laid out [Bn, sum_m a_size(m)])
// into the pool. `arrays` gets one CuArr per modality; `cublas_views` gets a
// rank-3-permuted view for rank-3 modalities (default-constructed slot for
// others so indices line up with `arrays`).
void pack_a_modalities(const Layout& L, int Bn, const float* src, CuPool* pool, CuArrVec* arrays,
                       CuArrVec* cublas_views, std::vector<float>* pack_scratch, std::vector<float>* cublas_scratch) {
  const int M = static_cast<int>(L.M);
  arrays->reserve(M);
  cublas_views->reserve(M);
  for (int m = 0; m < M; ++m) {
    const size_t per_batch = a_size(L, m);
    arrays->push_back(append_packed_slices(pool, pack_scratch, src, Bn, L.A_off[L.M], L.A_off[m], per_batch));
    const DependencyView deps = modality_state_deps(L, m);
    if (deps.rank == 3) {
      const int O_m     = static_cast<int>(L.O[m]);
      const int S_split = dep_state_size(L, deps, 2);
      const int S0      = dep_state_size(L, deps, 0);
      const int S1      = dep_state_size(L, deps, 1);
      cublas_views->push_back(
          append_rank3_cublas_view(pool, *pack_scratch, Bn, O_m, S0 * S1, S_split, per_batch, cublas_scratch));
    } else {
      cublas_views->emplace_back();
    }
  }
}

// Pack per-factor B-shaped data from `src` (laid out [Bn, sum_f b_size(f)]).
void pack_b_factors(const Layout& L, int Bn, const float* src, CuPool* pool, CuArrVec* arrays,
                    std::vector<float>* pack_scratch) {
  const int F = static_cast<int>(L.F);
  arrays->reserve(F);
  for (int f = 0; f < F; ++f) {
    arrays->push_back(append_packed_slices(pool, pack_scratch, src, Bn, L.B_off[L.F], L.B_off[f], b_size(L, f)));
  }
}

// Computes row-major out[b, m, n] = a_rm[b, m, k] @ q01_outer[b, k, n].
// The cuBLAS call is column-major, so the GEMM is expressed as C^T = Q^T @ A^T.
FfiError run_batched_q01_gemm(const StageCtx& stage, const char* op, const float* q01_outer, const float* a_rm,
                              float* out_rm, int M_rm, int N_rm, int K_rm) {
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

// Linear cache sig: A layout + T + Bn. C_tag captures C's content, K_m
// follows A — both are subsumed by A_sig.
inline uint64_t cuda_linear_sig(const Layout& L, int Bn) {
  return mix64(a_sig_bn(L, Bn), static_cast<uint64_t>(L.T));
}

// -----------------------------------------------------------------------------
// Per-call cache fills. prepare_caches() drives them in dependency order;
// each helper short-circuits on a content-tag + layout-sig hit.
// -----------------------------------------------------------------------------

// Per-(t, f) parent/action history bytes (without final-scatter or metadata).
inline size_t factor_history_pool_bytes(const FactorHistoryLevels& ftree, int T, int F) {
  size_t total = 0;
  for (int t = 0; t < T; ++t) {
    for (int f = 0; f < F; ++f) {
      const FactorHistoryLevel& lv = ftree[t][f];
      total += round_up_8(static_cast<size_t>(lv.n_histories) * lv.n_parents * sizeof(int32_t));
      total += round_up_8(static_cast<size_t>(lv.n_histories) * sizeof(int32_t));
    }
  }
  return total;
}

// Total bytes for the tree cache's pool: history grids + four final-scatter
// tables + four per-factor [F] int32 metadata arrays.
inline size_t tree_pool_bytes(const FactorHistoryLevels& ftree, const FactorHistoryTables& tables, int T, int F) {
  size_t total = factor_history_pool_bytes(ftree, T, F);
  total += round_up_8(tables.policy_to_modality_idx.size() * sizeof(int32_t));
  total += round_up_8(tables.factor_policy_to_history.size() * sizeof(int32_t));
  total += round_up_8(tables.mod_score_offsets.size() * sizeof(int64_t));
  total += round_up_8(tables.factor_score_offsets.size() * sizeof(int64_t));
  total += 4 * round_up_8(static_cast<size_t>(F) * sizeof(int32_t));
  return total;
}

// Copy each per-(t, f) parent_history / action_per_history block into `pool`,
// returning two CuArrGrid2D[T][F] grids. CuArr is move-only so the outer/
// inner vectors are resized in two passes rather than constructor-initialized.
void pack_factor_history_grids(const FactorHistoryLevels& ftree, int T, int F, CuPool* pool, CuArrGrid2D* parent_views,
                               CuArrGrid2D* action_views) {
  parent_views->resize(T);
  action_views->resize(T);
  for (int t = 0; t < T; ++t) {
    (*parent_views)[t].resize(F);
    (*action_views)[t].resize(F);
    for (int f = 0; f < F; ++f) {
      const FactorHistoryLevel& lv   = ftree[t][f];
      const size_t              p_sz = static_cast<size_t>(lv.n_histories) * lv.n_parents * sizeof(int32_t);
      const size_t              a_sz = static_cast<size_t>(lv.n_histories) * sizeof(int32_t);
      (*parent_views)[t][f]          = pool->append_copy(lv.parent_histories.data(), p_sz);
      (*action_views)[t][f]          = pool->append_copy(lv.action_per_history.data(), a_sz);
    }
  }
}

// Per-factor metadata arrays consumed by the v_full kernel.
struct FactorMetaPack {
  CuArr   S_dev;
  CuArr   depth_dev;
  CuArr   qs_off_dev;
  CuArr   I_off_dev;
  int64_t I_per_batch = 0;
};

FactorMetaPack pack_factor_metadata(const Layout& L, int F, CuPool* pool) {
  std::vector<int32_t> S_arr(F), depth_arr(F), qs_off_arr(F), I_off_arr(F);
  FactorMetaPack       out;
  for (int f = 0; f < F; ++f) {
    S_arr[f]      = static_cast<int32_t>(L.S[f]);
    depth_arr[f]  = static_cast<int32_t>(L.I_depths[f]);
    qs_off_arr[f] = static_cast<int32_t>(L.qs_off[f]);
    I_off_arr[f]  = static_cast<int32_t>(L.I_off[f]);
    out.I_per_batch += static_cast<int64_t>(L.I_depths[f]) * L.S[f];
  }
  const size_t meta_bytes = static_cast<size_t>(F) * sizeof(int32_t);
  out.S_dev               = pool->append_copy(S_arr.data(), meta_bytes);
  out.depth_dev           = pool->append_copy(depth_arr.data(), meta_bytes);
  out.qs_off_dev          = pool->append_copy(qs_off_arr.data(), meta_bytes);
  out.I_off_dev           = pool->append_copy(I_off_arr.data(), meta_bytes);
  return out;
}

// True when all batches of pm_base hold the same per-batch payload.
inline bool pm_is_broadcast(const int32_t* pm_base, int Bn, int P, int T, int F) {
  const size_t per_batch_bytes = static_cast<size_t>(P) * T * F * sizeof(int32_t);
  for (int b = 1; b < Bn; ++b) {
    if (std::memcmp(pm_base + static_cast<size_t>(b) * P * T * F, pm_base, per_batch_bytes) != 0) return false;
  }
  return true;
}

// Tree cache: per-(t, f) factor-history dedup + dependent index tables
// (mod_offs, ind_offs, mod_h_dims, pmi, p2h_concat, scratch sizings).
// Broadcast precondition (pm equal across batch) is enforced on miss.
FfiError fill_tree_cache(NegEfeContext& ctx, const Layout& L, int Bn, const int32_t* pm_base,
                         const void* pm_dev_src = nullptr) {
  const int      P              = static_cast<int>(L.P);
  const int      T              = static_cast<int>(L.T);
  const int      F              = static_cast<int>(L.F);
  const int64_t  pm_size_floats = static_cast<int64_t>(Bn) * P * T * F;
  const uint64_t pm_tag         = content_tag(reinterpret_cast<const float*>(pm_base), pm_size_floats);
  const uint64_t pm_sig         = factor_history_pm_sig(L);

  if (ctx.tree_cache.match_and_touch(pm_tag, pm_size_floats, pm_sig, pm_dev_src)) return FfiError::Success();

  if (!pm_is_broadcast(pm_base, Bn, P, T, F)) {
    return invalid_arg(kEfeKernelName, "CUDA path requires broadcast policy_matrix across batch");
  }

  FactorHistoryLevels ftree;
  PYMDP_TRY(build_factor_history_host(L, pm_base, &ftree));

  FactorHistoryTables tables;
  build_factor_history_tables(L, ftree, &tables, cuda_kernels::kRankMax, Bn);

  CuPool pool;
  CUDA_TRY("tree_pool_reserve", pool.reserve(tree_pool_bytes(ftree, tables, T, F)));

  CuArrGrid2D parent_views, action_views;
  pack_factor_history_grids(ftree, T, F, &pool, &parent_views, &action_views);

  CuArr pmi_dev =
      pool.append_copy(tables.policy_to_modality_idx.data(), tables.policy_to_modality_idx.size() * sizeof(int32_t));
  CuArr p2h_dev = pool.append_copy(tables.factor_policy_to_history.data(),
                                   tables.factor_policy_to_history.size() * sizeof(int32_t));
  CuArr mod_off_dev =
      pool.append_copy(tables.mod_score_offsets.data(), tables.mod_score_offsets.size() * sizeof(int64_t));
  CuArr ind_off_dev =
      pool.append_copy(tables.factor_score_offsets.data(), tables.factor_score_offsets.size() * sizeof(int64_t));

  FactorMetaPack meta = pack_factor_metadata(L, F, &pool);

  ctx.tree_cache.key.set(pm_tag, pm_size_floats, pm_sig, pm_dev_src);
  ctx.tree_cache.pool                         = std::move(pool);
  ctx.tree_cache.factor_tree                  = std::move(ftree);
  ctx.tree_cache.factor_parent_history        = std::move(parent_views);
  ctx.tree_cache.factor_action_per_history    = std::move(action_views);
  ctx.tree_cache.policy_to_modality_idx_dev   = std::move(pmi_dev);
  ctx.tree_cache.factor_policy_to_history_dev = std::move(p2h_dev);
  ctx.tree_cache.mod_score_offsets            = std::move(tables.mod_score_offsets);
  ctx.tree_cache.ind_score_offsets            = std::move(tables.factor_score_offsets);
  ctx.tree_cache.mod_score_offsets_dev        = std::move(mod_off_dev);
  ctx.tree_cache.ind_score_offsets_dev        = std::move(ind_off_dev);
  ctx.tree_cache.total_mod_entries            = tables.total_mod_entries;
  ctx.tree_cache.total_ind_entries            = tables.total_factor_entries;
  ctx.tree_cache.H_max_per_factor_max         = tables.max_history_count;
  ctx.tree_cache.mod_h_dims                   = std::move(tables.mod_h_dims);
  ctx.tree_cache.modality_tmp_qo_max_floats   = tables.modality_tmp_qo_max_floats;
  ctx.tree_cache.split_tmp_lin_max_floats     = tables.split_tmp_lin_max_floats;
  ctx.tree_cache.q01_outer_max_floats         = tables.q01_outer_max_floats;
  ctx.tree_cache.factor_S_dev                 = std::move(meta.S_dev);
  ctx.tree_cache.factor_depth_dev             = std::move(meta.depth_dev);
  ctx.tree_cache.factor_qs_off_dev            = std::move(meta.qs_off_dev);
  ctx.tree_cache.factor_I_off_dev             = std::move(meta.I_off_dev);
  ctx.tree_cache.I_per_batch                  = meta.I_per_batch;
  return FfiError::Success();
}

// A cache: per-modality [Bn, O_m, K_m] + cuBLAS-permuted view for rank-3.
// Returns A_tag for the downstream linear-cache key.
FfiError fill_a_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* A_base, uint64_t* A_tag_out,
                      const void* A_dev_src = nullptr) {
  const int64_t  A_total_size = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t A_tag        = content_tag(A_base, A_total_size);
  const uint64_t A_sig        = a_sig_bn(L, Bn);
  *A_tag_out                  = A_tag;

  if (ctx.a_cache.match_and_touch(A_tag, A_total_size, A_sig, A_dev_src)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("a_pool_reserve", pool.reserve(a_pack_pool_bytes(L, Bn)));
  CuArrVec arrays, cublas_views;
  pack_a_modalities(L, Bn, A_base, &pool, &arrays, &cublas_views, &g_cuda_host_pack_scratch,
                    &g_cuda_host_pack_scratch_alt);
  ctx.a_cache.store(A_tag, A_total_size, A_sig, std::move(pool), std::move(arrays), std::move(cublas_views), A_dev_src);
  return FfiError::Success();
}

// B cache: per-factor [Bn, S_f, K_f, U_f]; K_f = product of B-dep state sizes.
FfiError fill_b_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* B_base,
                      const void* B_dev_src = nullptr) {
  const int64_t  B_total_size = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t B_tag        = content_tag(B_base, B_total_size);
  const uint64_t B_sig        = b_sig_bn(L, Bn);

  if (ctx.b_cache.match_and_touch(B_tag, B_total_size, B_sig, B_dev_src)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("b_pool_reserve", pool.reserve(b_pack_pool_bytes(L, Bn)));
  CuArrVec arrays;
  pack_b_factors(L, Bn, B_base, &pool, &arrays, &g_cuda_host_pack_scratch);
  ctx.b_cache.store(B_tag, B_total_size, B_sig, std::move(pool), std::move(arrays), B_dev_src);
  return FfiError::Success();
}

FfiError fill_wa_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* pA_base, bool pA_present,
                       const void* pA_dev_src = nullptr) {
  if (!pA_present) {
    ctx.wa_cache.clear();
    return FfiError::Success();
  }
  const int64_t  A_total_size = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t pA_tag       = content_tag(pA_base, A_total_size);
  const uint64_t pA_sig       = a_sig_bn(L, Bn);
  if (ctx.wa_cache.match_and_touch(pA_tag, A_total_size, pA_sig, pA_dev_src)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("wa_pool_reserve", pool.reserve(a_pack_pool_bytes(L, Bn)));

  // Precompute wA per batch into a contiguous [Bn, A_total] buffer, then
  // hand it to pack_a_modalities just like raw A. `_alt` holds the cross-
  // batch precompute, so the rank-3 cuBLAS permute uses a local scratch
  // (only touched when at least one modality is rank-3).
  std::vector<float>& all_wA = g_cuda_host_pack_scratch_alt;
  all_wA.assign(static_cast<size_t>(Bn) * L.A_off[L.M], 0.0f);
  for (int b = 0; b < Bn; ++b) {
    const std::vector<float> wA_b = precompute_wA(L, pA_base + static_cast<int64_t>(b) * L.A_off[L.M]);
    std::memcpy(all_wA.data() + static_cast<size_t>(b) * L.A_off[L.M], wA_b.data(), wA_b.size() * sizeof(float));
  }

  CuArrVec           arrays, cublas_views;
  std::vector<float> cublas_scratch;
  pack_a_modalities(L, Bn, all_wA.data(), &pool, &arrays, &cublas_views, &g_cuda_host_pack_scratch, &cublas_scratch);
  ctx.wa_cache.store(pA_tag, A_total_size, pA_sig, std::move(pool), std::move(arrays), std::move(cublas_views),
                     pA_dev_src);
  return FfiError::Success();
}

FfiError fill_wb_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* pB_base, bool pB_present,
                       const void* pB_dev_src = nullptr) {
  if (!pB_present) {
    ctx.wb_cache.clear();
    return FfiError::Success();
  }
  const int64_t  B_total_size = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t pB_tag       = content_tag(pB_base, B_total_size);
  const uint64_t pB_sig       = b_sig_bn(L, Bn);
  if (ctx.wb_cache.match_and_touch(pB_tag, B_total_size, pB_sig, pB_dev_src)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("wb_pool_reserve", pool.reserve(b_pack_pool_bytes(L, Bn)));

  // Precompute wB (transposed layout) per batch, then hand the [Bn, B_total]
  // buffer to pack_b_factors.
  std::vector<float>& all_wB = g_cuda_host_pack_scratch_alt;
  all_wB.assign(static_cast<size_t>(Bn) * L.B_off[L.F], 0.0f);
  for (int b = 0; b < Bn; ++b) {
    const TransposedB wB_b = precompute_wB_transposed(L, pB_base + static_cast<int64_t>(b) * L.B_off[L.F]);
    std::memcpy(all_wB.data() + static_cast<size_t>(b) * L.B_off[L.F], wB_b.data.data(),
                wB_b.data.size() * sizeof(float));
  }

  CuArrVec arrays;
  pack_b_factors(L, Bn, all_wB.data(), &pool, &arrays, &g_cuda_host_pack_scratch);
  ctx.wb_cache.store(pB_tag, B_total_size, pB_sig, std::move(pool), std::move(arrays), pB_dev_src);
  return FfiError::Success();
}

// Bytes needed in a CuPool for the per-(t, m) linear cache.
inline size_t linear_pool_bytes(const Layout& L, int Bn) {
  const int T     = static_cast<int>(L.T);
  size_t    total = 0;
  for (int m = 0; m < static_cast<int>(L.M); ++m) {
    const int O   = static_cast<int>(L.O[m]);
    const int K_m = static_cast<int>(a_size(L, m) / O);
    total += static_cast<size_t>(T) * round_up_8(static_cast<size_t>(Bn) * K_m * sizeof(float));
  }
  return total;
}

// Build the per-(t, m) `linear` slice into `tmp` (size [Bn, K_m]):
//   linear[b, k] = (use_utility ? sum_o A[b, o, k] * C[b, t, m, o] : 0)
//                - (use_states_info_gain ? HA[b, m, k] : 0)
// HA = -sum_o xlogx(A); linear wants -HA, hence the subtract.
void build_linear_tm_slice(const Layout& L, int Bn, int t, int m, KernelFlags flags, const float* A_base,
                           const float* C_base, const std::vector<PrecomputedHA>& HA_per_batch,
                           std::vector<float>* tmp) {
  const int O   = static_cast<int>(L.O[m]);
  const int K_m = static_cast<int>(a_size(L, m) / O);
  tmp->assign(static_cast<size_t>(Bn) * K_m, 0.0f);
  for (int b = 0; b < Bn; ++b) {
    float*       dst  = tmp->data() + static_cast<size_t>(b) * K_m;
    const float* A_bm = A_base + b * L.A_off[L.M] + L.A_off[m];  // [O, K_m]
    if (flags.use_states_info_gain) {
      const PrecomputedHA& ha  = HA_per_batch[b];
      const float*         HAm = ha.data.data() + ha.offsets[m];
      for (int k = 0; k < K_m; ++k) dst[k] -= HAm[k];
    }
    if (flags.use_utility) {
      // C is [Bn, sum_m T*O_m]; the per-(t, m, b) slice is [O_m].
      const float* C_btm = C_base + b * L.C_off[L.M] + L.C_off[m] + t * O;
      for (int o = 0; o < O; ++o) {
        const float  c_o  = C_btm[o];
        const float* A_bo = A_bm + static_cast<size_t>(o) * K_m;
        for (int k = 0; k < K_m; ++k) dst[k] += A_bo[k] * c_o;
      }
    }
  }
}

// Linear cache: per-(t, m) [Bn, K_m] tensor fusing the HA entropy term and
// the A^T C utility term. Key uses only the two flags that shape `linear`.
FfiError fill_linear_cache(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, const float* A_base,
                           const float* C_base, uint64_t A_tag, const void* A_dev_src = nullptr,
                           const void* C_dev_src = nullptr) {
  const int      T             = static_cast<int>(L.T);
  const int      M             = static_cast<int>(L.M);
  const int64_t  C_total_size  = static_cast<int64_t>(Bn) * L.C_off[L.M];
  const uint64_t C_tag         = flags.use_utility ? content_tag(C_base, C_total_size) : 0;
  const uint64_t linear_sig    = cuda_linear_sig(L, Bn);
  const int32_t  flag_bits     = (flags.use_states_info_gain ? 1 : 0) | (flags.use_utility ? 2 : 0);
  const void*    c_dev_for_key = flags.use_utility ? C_dev_src : nullptr;
  if (ctx.linear_cache.match_and_touch(A_tag, C_tag, linear_sig, flag_bits, A_dev_src, c_dev_for_key)) {
    return FfiError::Success();
  }
  if (flag_bits == 0) {
    ctx.linear_cache.store_empty(A_tag, C_tag, linear_sig, flag_bits, A_dev_src, c_dev_for_key);
    return FfiError::Success();
  }

  std::vector<PrecomputedHA> HA_per_batch;
  if (flags.use_states_info_gain) {
    HA_per_batch.reserve(Bn);
    for (int b = 0; b < Bn; ++b) HA_per_batch.push_back(precompute_HA(L, A_base + b * L.A_off[L.M]));
  }

  CuPool pool;
  CUDA_TRY("linear_pool_reserve", pool.reserve(linear_pool_bytes(L, Bn)));

  CuArrGrid2D         per_tm(T);
  std::vector<float>& tmp = g_cuda_host_pack_scratch;
  for (int t = 0; t < T; ++t) {
    per_tm[t].resize(M);
    for (int m = 0; m < M; ++m) {
      build_linear_tm_slice(L, Bn, t, m, flags, A_base, C_base, HA_per_batch, &tmp);
      per_tm[t][m] = pool.append_copy(tmp.data(), tmp.size() * sizeof(float));
    }
  }

  ctx.linear_cache.a_tag         = A_tag;
  ctx.linear_cache.c_tag         = C_tag;
  ctx.linear_cache.layout_sig    = linear_sig;
  ctx.linear_cache.flags         = flag_bits;
  ctx.linear_cache.valid         = true;
  ctx.linear_cache.last_a_devptr = A_dev_src;
  ctx.linear_cache.last_c_devptr = c_dev_for_key;
  ctx.linear_cache.pool          = std::move(pool);
  ctx.linear_cache.per_tm        = std::move(per_tm);
  return FfiError::Success();
}

FfiError prepare_caches(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, const int32_t* pm_base,
                        const float* A_base, const float* B_base, const float* C_base, const float* pA_base,
                        const float* pB_base, bool pA_present, bool pB_present, DevSrcs srcs = {}) {
  PYMDP_TRY(fill_tree_cache(ctx, L, Bn, pm_base, srcs.pm));
  uint64_t A_tag = 0;
  PYMDP_TRY(fill_a_cache(ctx, L, Bn, A_base, &A_tag, srcs.A));
  PYMDP_TRY(fill_b_cache(ctx, L, Bn, B_base, srcs.B));
  if (flags.use_param_info_gain) {
    PYMDP_TRY(fill_wa_cache(ctx, L, Bn, pA_base, pA_present, srcs.pA));
    PYMDP_TRY(fill_wb_cache(ctx, L, Bn, pB_base, pB_present, srcs.pB));
  } else {
    ctx.wa_cache.clear();
    ctx.wb_cache.clear();
  }
  PYMDP_TRY(fill_linear_cache(ctx, L, flags, Bn, A_base, C_base, A_tag, srcs.A, srcs.C));
  return FfiError::Success();
}

// -----------------------------------------------------------------------------
// Forward pass
// -----------------------------------------------------------------------------

struct ForwardDims {
  int P;
  int T;
  int F;
  int M;
  int qsf;
};

ForwardDims forward_dims(const Layout& L) {
  return {static_cast<int>(L.P), static_cast<int>(L.T), static_cast<int>(L.F), static_cast<int>(L.M),
          static_cast<int>(L.qs_flat)};
}

FfiError ensure_forward_scratch(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, cudaStream_t stream,
                                bool out_is_external) {
  const ForwardDims          d                 = forward_dims(L);
  const FactorHistoryLevels& ftree             = ctx.tree_cache.factor_tree;
  const int64_t              total_mod_entries = ctx.tree_cache.total_mod_entries;
  const int64_t              total_ind_entries = ctx.tree_cache.total_ind_entries;

  // resize is a no-op when size already matches; CuArr::ensure grows lazily.
  ctx.scratch.qs_init_per_factor.resize(d.F);
  for (int slot = 0; slot < 2; ++slot) {
    ctx.scratch.qs_factor_buf[slot].resize(d.F);
  }
  for (int f = 0; f < d.F; ++f) {
    const int Sf = static_cast<int>(L.S[f]);
    CUDA_TRY("scratch qs_init", ctx.scratch.qs_init_per_factor[f].ensure(static_cast<size_t>(Bn) * Sf * sizeof(float)));
    const int    H_f_max      = static_cast<int>(max_factor_histories_for_factor(ftree, d.T, f));
    const size_t qs_buf_bytes = static_cast<size_t>(Bn) * H_f_max * Sf * sizeof(float);
    for (int slot = 0; slot < 2; ++slot) {
      CUDA_TRY("scratch qs_factor", ctx.scratch.qs_factor_buf[slot][f].ensure(qs_buf_bytes));
    }
  }
  CUDA_TRY("scratch scores_concat",
           ctx.scratch.scores_concat.ensure(static_cast<size_t>(Bn) * total_mod_entries * sizeof(float)));
  // out_is_external: the dev path scatters directly into JAX's output buffer
  // (threaded through StageCtx.out_dev_ptr), so we must not allocate or
  // ensure() the owned scratch slot.
  if (!out_is_external) {
    CUDA_TRY("scratch out_dev", ctx.scratch.out_dev.ensure(static_cast<size_t>(Bn) * d.P * sizeof(float)));
  }
  if (flags.use_inductive) {
    CUDA_TRY("scratch v_full", ctx.scratch.v_full_dev.ensure(static_cast<size_t>(Bn) * d.qsf * sizeof(float)));
    CUDA_TRY("scratch inductive_concat",
             ctx.scratch.inductive_concat.ensure(static_cast<size_t>(Bn) * total_ind_entries * sizeof(float)));
  }
  if (needs_factor_scores(ctx, flags)) {
    CUDA_TRY("scratch factor_scores",
             ctx.scratch.factor_scores.ensure(static_cast<size_t>(Bn) * total_ind_entries * sizeof(float)));
  }
  if (ctx.tree_cache.modality_tmp_qo_max_floats > 0) {
    CUDA_TRY("scratch split_tmp_qo",
             ctx.scratch.split_tmp_qo.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    if (flags.use_param_info_gain) {
      CUDA_TRY("scratch split_tmp_wa",
               ctx.scratch.split_tmp_wa.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
      CUDA_TRY("scratch tmp_wa_cublas",
               ctx.scratch.tmp_wa_cublas.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    }
    CUDA_TRY("scratch split_tmp_lin",
             ctx.scratch.split_tmp_lin.ensure(ctx.tree_cache.split_tmp_lin_max_floats * sizeof(float)));
    // Both q01_outer and tmp_qo_cublas are sized for the worst-case rank-2/3
    // modality (cache-fill pass takes the max across modalities).
    CUDA_TRY("scratch q01_outer", ctx.scratch.q01_outer.ensure(ctx.tree_cache.q01_outer_max_floats * sizeof(float)));
    CUDA_TRY("scratch tmp_qo_cublas",
             ctx.scratch.tmp_qo_cublas.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    PYMDP_TRY(ensure_cublas_handle(ctx));
    // Skip the cublasSetStream syscall when JAX hands us the same stream as
    // last call — the JIT typically keeps the stream stable across invocations
    // of the same compiled executable, so this short-circuits to a pointer
    // compare on the steady-state path.
    if (ctx.cublas_handle.bound_stream != stream) {
      CUBLAS_TRY("cublasSetStream", cublasSetStream(ctx.cublas_handle.handle, stream));
      ctx.cublas_handle.bound_stream = stream;
    }
  }
  return FfiError::Success();
}

void upload_qs_init(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base) {
  const ForwardDims   d   = forward_dims(L);
  std::vector<float>& tmp = g_cuda_host_pack_scratch;
  for (int f = 0; f < d.F; ++f) {
    const size_t Sf = static_cast<size_t>(L.S[f]);
    pack_batched_slices(&tmp, qs_base, Bn, d.qsf, L.qs_off[f], Sf);
    std::memcpy(ctx.scratch.qs_init_per_factor[f].ptr, tmp.data(), static_cast<size_t>(Bn) * Sf * sizeof(float));
  }
}

// D2D variant of upload_qs_init for the dev path: one cudaMemcpy2DAsync per
// factor, gathering [Bn, qs_flat] strided into contiguous per-factor [Bn, S_f].
// Stays on the XLA stream — kernel reads downstream preserve ordering.
FfiError upload_qs_init_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev, cudaStream_t stream) {
  const ForwardDims d = forward_dims(L);
  for (int f = 0; f < d.F; ++f) {
    const size_t Sf = static_cast<size_t>(L.S[f]);
    CUDA_TRY("qs_init_d2d",
             cudaMemcpy2DAsync(ctx.scratch.qs_init_per_factor[f].ptr,       // dst
                               Sf * sizeof(float),                          // dst_pitch
                               qs_dev + L.qs_off[f],                        // src start
                               static_cast<size_t>(d.qsf) * sizeof(float),  // src_pitch
                               Sf * sizeof(float),                          // width_bytes
                               Bn,                                          // height (rows)
                               cudaMemcpyDeviceToDevice, stream));
  }
  return FfiError::Success();
}

// D2D variant of upload_inductive_vector: launches the v_full kernel
// directly against JAX's qs_init / I / eps device pointers (no D2H, no
// host argmax). Requires the tree cache to be filled (per-factor metadata).
FfiError upload_inductive_vector_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev,
                                     const float* I_dev, const float* eps_dev, EpsilonLayout epsilon,
                                     cudaStream_t stream) {
  const ForwardDims d          = forward_dims(L);
  const int         eps_stride = epsilon.batched ? 1 : 0;
  CUDA_TRY("v_full",
           cuda_kernels::launch_v_full(
               qs_dev, I_dev, eps_dev, eps_stride, Bn, d.F, d.qsf, ctx.tree_cache.I_per_batch,
               ctx.tree_cache.factor_S_dev.as<const int32_t>(), ctx.tree_cache.factor_depth_dev.as<const int32_t>(),
               ctx.tree_cache.factor_qs_off_dev.as<const int32_t>(),
               ctx.tree_cache.factor_I_off_dev.as<const int32_t>(), ctx.scratch.v_full_dev.as<float>(), stream));
  return FfiError::Success();
}

FfiError upload_inductive_vector(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base, const float* I_base,
                                 const float* eps_base, EpsilonLayout epsilon) {
  const ForwardDims   d        = forward_dims(L);
  const size_t        n_packed = static_cast<size_t>(Bn) * d.qsf;
  std::vector<float>& v_packed = g_cuda_host_pack_scratch;
  ensure_at_least(v_packed, n_packed);
  for (int b = 0; b < Bn; ++b) {
    const float e = eps_base[epsilon.batched ? b : 0];
    PYMDP_TRY(check_inductive_epsilon_value(e));
    const PrecomputedInductive v = precompute_inductive(L, qs_base + b * d.qsf, I_base + b * L.I_off[L.F], e);
    std::memcpy(v_packed.data() + b * d.qsf, v.data.data(), d.qsf * sizeof(float));
  }
  std::memcpy(ctx.scratch.v_full_dev.ptr, v_packed.data(), n_packed * sizeof(float));
  return FfiError::Success();
}

struct QsLevel {
  std::vector<const float*> ptrs;
  std::vector<int>          histories;
};

QsLevel initial_qs_level(const StageCtx& stage) {
  const int F = static_cast<int>(stage.L.F);
  QsLevel   level;
  level.ptrs.resize(F);
  level.histories.assign(F, 1);
  for (int f = 0; f < F; ++f) {
    level.ptrs[f] = stage.ctx.scratch.qs_init_per_factor[f].as<const float>();
  }
  return level;
}

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
    const bool           do_fs    = needs_factor_scores(ctx, stage.flags);
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

FfiError launch_modality_rank2(const StageCtx& stage, const QsLevel& qs_next, const ModalityLaunch& ml,
                               int64_t total_mod_entries) {
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

  PYMDP_TRY(build_q01_outer_for_modality(stage, qs_next, ml, q01_outer));
  if (ml.use_states || ml.use_pA) {
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 modality GEMM", q01_outer, ml.A_unflat, tmp_qo_cb, ml.O, H_kk, K_d));
  }
  if (ml.use_pA) {
    PYMDP_TRY(run_batched_q01_gemm(stage, "rank-2 pA GEMM", q01_outer, ml.wA_unflat, tmp_wa_cb, ml.O, H_kk, K_d));
  }
  CUDA_TRY("modality_score_dedup_rank2_cublas_finish",
           cuda_kernels::launch_modality_score_dedup_rank2_cublas_finish(
               tmp_qo_cb, tmp_wa_cb, q01_outer, ml.linear, stage.Bn, ml.O, H_kk, K_d, total_mod_entries, ml.use_states,
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
    // cuBLAS sgemm beats the per-(b,h,s) thread-K kernel above ~H_kk=32 on
    // sm_53 (microbench_modality_subkernels); dispatch overhead loses below.
    constexpr int kTmpLinCublasMinHkk = 32;
    if (H_kk >= kTmpLinCublasMinHkk) {
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

FfiError launch_modality_scores(const StageCtx& stage, int t, const QsLevel& qs_next) {
  const int     M                 = static_cast<int>(stage.L.M);
  const int64_t total_mod_entries = stage.ctx.tree_cache.total_mod_entries;
  for (int m = 0; m < M; ++m) {
    const ModalityLaunch ml = make_modality_launch(stage, t, m);
    switch (ml.deps.rank) {
    case 1:
      PYMDP_TRY(launch_modality_rank1(stage, qs_next, ml, total_mod_entries));
      break;
    case 2:
      PYMDP_TRY(launch_modality_rank2(stage, qs_next, ml, total_mod_entries));
      break;
    case 3:
      PYMDP_TRY(launch_modality_rank3(stage, qs_next, ml, m, total_mod_entries));
      break;
    default:
      return invalid_arg(kEfeKernelName, "CUDA path supports modality rank in [1, 3]");
    }
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

// Drives the forward pass. Caller must have called ensure_forward_scratch,
// upload_qs_init[_d2d], and (when use_inductive) upload_inductive_vector[_d2d].
// scatter_out_dev: device-side scatter destination (host path: ctx.scratch
// .out_dev managed buffer; dev path: JAX's output device buffer).
// out_host_or_null: non-null (NegEfeCudaHost) → D2H from scatter_out_dev
// into the host out buffer + sync after scatter; nullptr (NegEfeCudaDev)
// → scatter wrote straight to JAX's output, no D2H needed.
FfiError run_forward(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, float* scatter_out_dev,
                     float* out_host_or_null, cudaStream_t stream = nullptr) {
  const StageCtx    stage{ctx, L, flags, Bn, stream, scatter_out_dev};
  const ForwardDims d = forward_dims(L);

  // launch_b_rollout_level folds the per-(t, f) inductive score into phase 2
  // when use_inductive is on, so there is no separate inductive launch step.
  QsLevel curr = initial_qs_level(stage);
  for (int t = 0; t < d.T; ++t) {
    QsLevel next;
    PYMDP_TRY(launch_b_rollout_level(stage, t, curr, &next));
    PYMDP_TRY(launch_modality_scores(stage, t, next));
    curr = std::move(next);
  }

  PYMDP_TRY(launch_final_scatter(stage));
  if (out_host_or_null != nullptr) {
    // D2H into host out buffer + sync (Tegra: managed memory aliases host DRAM).
    CUDA_TRY("memcpy_async_out",
             cudaMemcpyAsync(out_host_or_null, scatter_out_dev, static_cast<size_t>(Bn) * d.P * sizeof(float),
                             cudaMemcpyDefault, stream));
    CUDA_TRY("stream_sync", cudaStreamSynchronize(stream));
  }
  return FfiError::Success();
}

// -----------------------------------------------------------------------------
// Entry-point glue: warm-cache check, async D2H. The parse + validate prologue
// (ParsedCall / parse_and_validate_call) is shared via neg_efe_entry.h.
// -----------------------------------------------------------------------------

// True when all six caches still match the current devptrs / sigs.
bool caches_are_warm(const NegEfeContext& ctx, const Layout& L, int Bn, KernelFlags flags, bool pA_present,
                     bool pB_present, const DevSrcs& devs) {
  const uint64_t pm_sig       = factor_history_pm_sig(L);
  const int64_t  pm_size      = static_cast<int64_t>(Bn) * L.P * L.T * L.F;
  const uint64_t a_sig        = a_sig_bn(L, Bn);
  const int64_t  a_size       = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t b_sig        = b_sig_bn(L, Bn);
  const int64_t  b_size       = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t linear_sig   = cuda_linear_sig(L, Bn);
  const int32_t  linear_flags = (flags.use_states_info_gain ? 1 : 0) | (flags.use_utility ? 2 : 0);

  if (!ctx.tree_cache.match_devptr(devs.pm, pm_size, pm_sig)) return false;
  if (!ctx.a_cache.match_devptr(devs.A, a_size, a_sig)) return false;
  if (!ctx.b_cache.match_devptr(devs.B, b_size, b_sig)) return false;
  if (!ctx.linear_cache.match_devptr(devs.A, devs.C, linear_sig, linear_flags)) return false;
  if (flags.use_param_info_gain && pA_present && !ctx.wa_cache.match_devptr(devs.pA, a_size, a_sig)) return false;
  if (flags.use_param_info_gain && pB_present && !ctx.wb_cache.match_devptr(devs.pB, b_size, b_sig)) return false;
  return true;
}

// Async D2H from a device buffer into a host vector. Caller syncs the stream
// before reading the host bytes.
template <typename T> FfiError d2h_into(std::vector<T>* dst, const T* src_dev, std::size_t n, cudaStream_t stream) {
  dst->resize(n);
  if (n == 0) return FfiError::Success();
  cudaError_t rc = cudaMemcpyAsync(dst->data(), src_dev, n * sizeof(T), cudaMemcpyDeviceToHost, stream);
  if (rc != cudaSuccess) {
    return invalid_arg(kEfeKernelName, std::string("cudaMemcpyAsync D2H failed: ") + cudaGetErrorString(rc));
  }
  return FfiError::Success();
}

// Tegra zero-copy variant of d2h_into. If `src_dev` is host-accessible
// (managed or pinned), returns the host alias and skips the D2H entirely.
// Otherwise queues the cudaMemcpyAsync into `dst` and returns dst->data().
// Caller still must sync the stream before reading the result, regardless of
// path: even for managed/pinned memory, prior queued ops may still be writing
// to it.
template <typename T>
FfiError d2h_or_alias(std::vector<T>* dst, const T* src_dev, std::size_t n, cudaStream_t stream, const T** out_host) {
  if (n == 0) {
    *out_host = nullptr;
    return FfiError::Success();
  }
  if (void* alias = try_alias_as_host(src_dev)) {
    *out_host = static_cast<const T*>(alias);
    return FfiError::Success();
  }
  PYMDP_TRY(d2h_into(dst, src_dev, n, stream));
  *out_host = dst->data();
  return FfiError::Success();
}

void log_efe_cuda_timestats_if_enabled(NegEfeContext& ctx, const Layout& L, int64_t Bn) {
  if (const char* env = std::getenv("PYMDP_FFI_TIME"); !env || env[0] != '1') return;
  int            device = -1;
  cudaError_t    rc     = cudaGetDevice(&device);
  cudaDeviceProp prop{};
  if (rc == cudaSuccess) cudaGetDeviceProperties(&prop, device);
  int64_t factor_history_total = 0;
  for (const FactorHistoryRow& tlevel : ctx.tree_cache.factor_tree) {
    for (const FactorHistoryLevel& lv : tlevel) factor_history_total += lv.n_histories;
  }
  int64_t linear_tm_total = 0;
  for (const CuArrVec& row : ctx.linear_cache.per_tm) linear_tm_total += row.size();
  std::fprintf(stderr,
               "[efe-cuda] device=%d sm_%d%d Bn=%lld P=%lld T=%lld | factor_histories=%lld "
               "(policy_steps=%lld) | A_cached=%d A_cublas_cached=%d B_cached=%d linear_tm=%lld\n",
               device, rc == cudaSuccess ? prop.major : 0, rc == cudaSuccess ? prop.minor : 0, (long long)Bn,
               (long long)L.P, (long long)L.T, (long long)factor_history_total, (long long)(L.P * L.T),
               static_cast<int>(ctx.a_cache.arrays.size()), static_cast<int>(ctx.a_cache.cublas_views.size()),
               static_cast<int>(ctx.b_cache.arrays.size()), (long long)linear_tm_total);
}

}  // namespace

// -----------------------------------------------------------------------------
// Entry points
// -----------------------------------------------------------------------------

// Host-buffer ABI (platform="cpu"). GPU work runs through managed scratch;
// scatter writes ctx.scratch.out_dev (owned) and D2H copies into the FFI
// out buffer at the end.

FfiError NegEfeCudaHost(NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B,
                        FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                        FfiF32Out out, FfiInt64Span S_span, FfiInt64Span O_span, FfiInt64Span U_span,
                        FfiInt64Span qs_off_span, FfiInt64Span A_off_span, FfiInt64Span B_off_span,
                        FfiInt64Span C_off_span, FfiInt64Span I_off_span, FfiInt64Span I_depths_span,
                        FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span, FfiInt64Span B_dep_flat_span,
                        FfiInt64Span B_dep_off_span, int32_t flags) {
  const LayoutSpans spans{S_span,         O_span,          U_span,        qs_off_span,   A_off_span,
                          B_off_span,     C_off_span,      I_off_span,    I_depths_span, A_dep_flat_span,
                          A_dep_off_span, B_dep_flat_span, B_dep_off_span};
  ParsedCall        pc;
  PYMDP_TRY(
      parse_and_validate_call(policy_matrix, qs_init, A, B, C, I, pA, pB, inductive_epsilon, out, spans, flags, &pc));

  NegEfeContext& ctx    = *state->ctx;
  const int      Bn_int = static_cast<int>(pc.shape.Bn);

  PYMDP_TRY(prepare_caches(ctx, pc.L, pc.flags, Bn_int, policy_matrix.typed_data(), A.typed_data(), B.typed_data(),
                           C.typed_data(), pA.typed_data(), pB.typed_data(), pc.pA_present, pc.pB_present));

  log_efe_cuda_timestats_if_enabled(ctx, pc.L, pc.shape.Bn);

  PYMDP_TRY(ensure_forward_scratch(ctx, pc.L, pc.flags, Bn_int, /*stream=*/nullptr, /*out_is_external=*/false));
  upload_qs_init(ctx, pc.L, Bn_int, qs_init.typed_data());
  if (pc.flags.use_inductive) {
    PYMDP_TRY(upload_inductive_vector(ctx, pc.L, Bn_int, qs_init.typed_data(), I.typed_data(),
                                      inductive_epsilon.typed_data(), pc.epsilon));
  }
  return run_forward(ctx, pc.L, pc.flags, Bn_int, ctx.scratch.out_dev.as<float>(), out->typed_data());
}

// Device-buffer ABI (platform="CUDA"). JAX passes device buffers + stream;
// scatter writes JAX's output directly. Warm path (all caches match the
// prior call's devptrs) skips D2H entirely; cold path does D2H + content-tag
// cache fill, then records the new devptrs.

ffi::TypeId NegEfeState::id = {};

NegEfeState::NegEfeState() : ctx(std::make_unique<NegEfeContext>()) {}
NegEfeState::~NegEfeState() = default;

ffi::ErrorOr<std::unique_ptr<NegEfeState>> NegEfeCudaInstantiate() {
  return std::make_unique<NegEfeState>();
}

FfiError NegEfeCudaDev(cudaStream_t stream, NegEfeState* state, FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A,
                       FfiF32Buf B, FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon,
                       FfiF32Out out, FfiInt64Span S_span, FfiInt64Span O_span, FfiInt64Span U_span,
                       FfiInt64Span qs_off_span, FfiInt64Span A_off_span, FfiInt64Span B_off_span,
                       FfiInt64Span C_off_span, FfiInt64Span I_off_span, FfiInt64Span I_depths_span,
                       FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span, FfiInt64Span B_dep_flat_span,
                       FfiInt64Span B_dep_off_span, int32_t flags) {
  const LayoutSpans spans{S_span,         O_span,          U_span,        qs_off_span,   A_off_span,
                          B_off_span,     C_off_span,      I_off_span,    I_depths_span, A_dep_flat_span,
                          A_dep_off_span, B_dep_flat_span, B_dep_off_span};
  ParsedCall        pc;
  PYMDP_TRY(
      parse_and_validate_call(policy_matrix, qs_init, A, B, C, I, pA, pB, inductive_epsilon, out, spans, flags, &pc));

  NegEfeContext& ctx    = *state->ctx;
  const int      Bn_int = static_cast<int>(pc.shape.Bn);

  const DevSrcs devs{
      .pm = policy_matrix.typed_data(),
      .A  = A.typed_data(),
      .B  = B.typed_data(),
      .C  = pc.flags.use_utility ? static_cast<const void*>(C.typed_data()) : nullptr,
      .pA = pc.pA_present ? static_cast<const void*>(pA.typed_data()) : nullptr,
      .pB = pc.pB_present ? static_cast<const void*>(pB.typed_data()) : nullptr,
  };

  // When param_info_gain is on but pA / pB is absent, the warm-check skips
  // the absent slot — clear any stale prior-call entry so downstream
  // `!empty()` guards read absence correctly.
  if (pc.flags.use_param_info_gain) {
    if (!pc.pA_present) ctx.wa_cache.clear();
    if (!pc.pB_present) ctx.wb_cache.clear();
  }

  if (!caches_are_warm(ctx, pc.L, Bn_int, pc.flags, pc.pA_present, pc.pB_present, devs)) {
    // Tegra zero-copy on the cold-cache path. For each input buffer try the
    // host alias first (managed/pinned memory has a host address); only D2H
    // when JAX hands us device-only memory. The stream sync below is still
    // required either way — prior queued ops may still be writing these
    // buffers.
    thread_local std::vector<int32_t> pm_host;
    thread_local std::vector<float>   A_host, B_host, C_host, pA_host, pB_host;
    const int32_t*                    pm_ptr = nullptr;
    const float*                      A_ptr  = nullptr;
    const float*                      B_ptr  = nullptr;
    const float*                      C_ptr  = nullptr;
    const float*                      pA_ptr = nullptr;
    const float*                      pB_ptr = nullptr;
    PYMDP_TRY(d2h_or_alias(&pm_host, policy_matrix.typed_data(), policy_matrix.element_count(), stream, &pm_ptr));
    PYMDP_TRY(d2h_or_alias(&A_host, A.typed_data(), A.element_count(), stream, &A_ptr));
    PYMDP_TRY(d2h_or_alias(&B_host, B.typed_data(), B.element_count(), stream, &B_ptr));
    if (pc.flags.use_utility) PYMDP_TRY(d2h_or_alias(&C_host, C.typed_data(), C.element_count(), stream, &C_ptr));
    if (pc.pA_present) PYMDP_TRY(d2h_or_alias(&pA_host, pA.typed_data(), pA.element_count(), stream, &pA_ptr));
    if (pc.pB_present) PYMDP_TRY(d2h_or_alias(&pB_host, pB.typed_data(), pB.element_count(), stream, &pB_ptr));
    if (cudaError_t rc = cudaStreamSynchronize(stream); rc != cudaSuccess) {
      return invalid_arg(kEfeKernelName,
                         std::string("cudaStreamSynchronize before host work failed: ") + cudaGetErrorString(rc));
    }
    PYMDP_TRY(prepare_caches(ctx, pc.L, pc.flags, Bn_int, pm_ptr, A_ptr, B_ptr, C_ptr, pA_ptr, pB_ptr, pc.pA_present,
                             pc.pB_present, devs));
  }

  // JAX's output device buffer flows straight through StageCtx.out_dev_ptr
  // into launch_final_scatter — never stored in a CuArr, so ensure() / reset()
  // can't free a JAX-owned pointer.
  if (static_cast<int64_t>(out->element_count()) != pc.shape.Bn * pc.shape.P) {
    return invalid_arg(kEfeKernelName, "out element_count " + std::to_string(out->element_count()) +
                                           " != Bn*P = " + std::to_string(pc.shape.Bn * pc.shape.P));
  }

  PYMDP_TRY(ensure_forward_scratch(ctx, pc.L, pc.flags, Bn_int, stream, /*out_is_external=*/true));
  PYMDP_TRY(upload_qs_init_d2d(ctx, pc.L, Bn_int, qs_init.typed_data(), stream));
  if (pc.flags.use_inductive) {
    PYMDP_TRY(upload_inductive_vector_d2d(ctx, pc.L, Bn_int, qs_init.typed_data(), I.typed_data(),
                                          inductive_epsilon.typed_data(), pc.epsilon, stream));
  }
  PYMDP_TRY(run_forward(ctx, pc.L, pc.flags, Bn_int, out->typed_data(), /*out_host_or_null=*/nullptr, stream));
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
