// Cache materialization for the CUDA neg-EFE forward pass. Each per-cache
// fill helper is keyed on (content tag, layout sig) and short-circuits on a
// match; on miss it stages host packing into a CuPool and stores the
// resulting CuArrs in NegEfeContext. The content tag is a sampled fingerprint
// of the host buffer, so reuse of a device address for mutated data is
// detected here on the cold path and by caches_are_warm (entry.cc) on the
// warm path.

#include "neg_efe/neg_efe.h"

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/cuda_memory.h"
#include "common/error_helpers.h"
#include "common/kernel_primitives.h"
#include "neg_efe/factor_history_tables.h"
#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_cuda_context.h"
#include "neg_efe/neg_efe_cuda_internal.h"
#include "neg_efe/neg_efe_cuda_kernels.h"
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"

// CUDA error category for this TU. Pinned to "efe_ffi" so FfiErrors raised
// inside CUDA_TRY surface as neg-EFE errors.
#define CUDA_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cuda_err(::pymdp_ffi::kEfeKernelName, op, (expr)))

namespace pymdp_ffi {
namespace {

// kRankMax: per-modality dependency-rank cap (mirrors neg_efe_cuda_kernels.h).
static_assert(cuda_kernels::kRankMax == 3, "kRankMax assumed = 3 by index encoding");

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

// -----------------------------------------------------------------------------
// Tree-cache packing helpers
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

// -----------------------------------------------------------------------------
// Per-cache fills. prepare_caches() drives them in dependency order; each
// helper short-circuits on a content-tag + layout-sig hit.
// -----------------------------------------------------------------------------

// Tree cache: per-(t, f) factor-history dedup + dependent index tables
// (mod_offs, ind_offs, mod_h_dims, pmi, p2h_concat, scratch sizings).
// Broadcast precondition (pm equal across batch) is enforced on miss.
FfiError fill_tree_cache(NegEfeContext& ctx, const Layout& L, int Bn, const int32_t* pm_base) {
  const int      P              = static_cast<int>(L.P);
  const int      T              = static_cast<int>(L.T);
  const int      F              = static_cast<int>(L.F);
  const int64_t  pm_size_floats = static_cast<int64_t>(Bn) * P * T * F;
  const uint64_t pm_tag         = content_tag(reinterpret_cast<const float*>(pm_base), pm_size_floats);
  const uint64_t pm_sig         = factor_history_pm_sig(L);

  if (ctx.tree_cache.match(pm_tag, pm_size_floats, pm_sig)) return FfiError::Success();

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

  ctx.tree_cache.key.set(pm_tag, pm_size_floats, pm_sig);
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
FfiError fill_a_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* A_base, uint64_t* A_tag_out) {
  const int64_t  A_total_size = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t A_tag        = content_tag(A_base, A_total_size);
  const uint64_t A_sig        = a_sig_bn(L, Bn);
  *A_tag_out                  = A_tag;

  if (ctx.a_cache.match(A_tag, A_total_size, A_sig)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("a_pool_reserve", pool.reserve(a_pack_pool_bytes(L, Bn)));
  CuArrVec arrays, cublas_views;
  pack_a_modalities(L, Bn, A_base, &pool, &arrays, &cublas_views, &g_cuda_host_pack_scratch,
                    &g_cuda_host_pack_scratch_alt);
  ctx.a_cache.store(A_tag, A_total_size, A_sig, std::move(pool), std::move(arrays), std::move(cublas_views));
  return FfiError::Success();
}

// B cache: per-factor [Bn, S_f, K_f, U_f]; K_f = product of B-dep state sizes.
FfiError fill_b_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* B_base) {
  const int64_t  B_total_size = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t B_tag        = content_tag(B_base, B_total_size);
  const uint64_t B_sig        = b_sig_bn(L, Bn);

  if (ctx.b_cache.match(B_tag, B_total_size, B_sig)) return FfiError::Success();

  CuPool pool;
  CUDA_TRY("b_pool_reserve", pool.reserve(b_pack_pool_bytes(L, Bn)));
  CuArrVec arrays;
  pack_b_factors(L, Bn, B_base, &pool, &arrays, &g_cuda_host_pack_scratch);
  ctx.b_cache.store(B_tag, B_total_size, B_sig, std::move(pool), std::move(arrays));
  return FfiError::Success();
}

FfiError fill_wa_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* pA_base, bool pA_present) {
  if (!pA_present) {
    ctx.wa_cache.clear();
    return FfiError::Success();
  }
  const int64_t  A_total_size = static_cast<int64_t>(Bn) * L.A_off[L.M];
  const uint64_t pA_tag       = content_tag(pA_base, A_total_size);
  const uint64_t pA_sig       = a_sig_bn(L, Bn);
  if (ctx.wa_cache.match(pA_tag, A_total_size, pA_sig)) return FfiError::Success();

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
  ctx.wa_cache.store(pA_tag, A_total_size, pA_sig, std::move(pool), std::move(arrays), std::move(cublas_views));
  return FfiError::Success();
}

FfiError fill_wb_cache(NegEfeContext& ctx, const Layout& L, int Bn, const float* pB_base, bool pB_present) {
  if (!pB_present) {
    ctx.wb_cache.clear();
    return FfiError::Success();
  }
  const int64_t  B_total_size = static_cast<int64_t>(Bn) * L.B_off[L.F];
  const uint64_t pB_tag       = content_tag(pB_base, B_total_size);
  const uint64_t pB_sig       = b_sig_bn(L, Bn);
  if (ctx.wb_cache.match(pB_tag, B_total_size, pB_sig)) return FfiError::Success();

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
  ctx.wb_cache.store(pB_tag, B_total_size, pB_sig, std::move(pool), std::move(arrays));
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
                           const float* C_base, uint64_t A_tag) {
  const int      T            = static_cast<int>(L.T);
  const int      M            = static_cast<int>(L.M);
  const int64_t  C_total_size = static_cast<int64_t>(Bn) * L.C_off[L.M];
  const uint64_t C_tag        = flags.use_utility ? content_tag(C_base, C_total_size) : 0;
  const uint64_t linear_sig   = cuda_linear_sig(L, Bn);
  const int32_t  flag_bits    = (flags.use_states_info_gain ? 1 : 0) | (flags.use_utility ? 2 : 0);
  if (ctx.linear_cache.match(A_tag, C_tag, linear_sig, flag_bits)) {
    return FfiError::Success();
  }
  if (flag_bits == 0) {
    ctx.linear_cache.store_empty(A_tag, C_tag, linear_sig, flag_bits);
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

  ctx.linear_cache.a_tag      = A_tag;
  ctx.linear_cache.c_tag      = C_tag;
  ctx.linear_cache.layout_sig = linear_sig;
  ctx.linear_cache.flags      = flag_bits;
  ctx.linear_cache.valid      = true;
  ctx.linear_cache.pool       = std::move(pool);
  ctx.linear_cache.per_tm     = std::move(per_tm);
  return FfiError::Success();
}

}  // namespace

FfiError prepare_caches(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, const int32_t* pm_base,
                        const float* A_base, const float* B_base, const float* C_base, const float* pA_base,
                        const float* pB_base, bool pA_present, bool pB_present) {
  PYMDP_TRY(fill_tree_cache(ctx, L, Bn, pm_base));
  uint64_t A_tag = 0;
  PYMDP_TRY(fill_a_cache(ctx, L, Bn, A_base, &A_tag));
  PYMDP_TRY(fill_b_cache(ctx, L, Bn, B_base));
  if (flags.use_param_info_gain) {
    PYMDP_TRY(fill_wa_cache(ctx, L, Bn, pA_base, pA_present));
    PYMDP_TRY(fill_wb_cache(ctx, L, Bn, pB_base, pB_present));
  } else {
    ctx.wa_cache.clear();
    ctx.wb_cache.clear();
  }
  PYMDP_TRY(fill_linear_cache(ctx, L, flags, Bn, A_base, C_base, A_tag));
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
