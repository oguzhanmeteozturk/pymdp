#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <unordered_map>
#include <vector>

#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_layout.h"

namespace pymdp_ffi {

inline int64_t factor_history_tm_index(int64_t t, int64_t m, int64_t M) {
  return t * M + m;
}

inline size_t factor_history_tm_rank_dim_index(int64_t t, int64_t m, int i, int64_t M, int rank_dim_cap) {
  return (static_cast<size_t>(t) * static_cast<size_t>(M) + static_cast<size_t>(m)) *
             static_cast<size_t>(rank_dim_cap) +
         static_cast<size_t>(i);
}

struct FactorHistoryTables {
  FfiInt64Vec mod_score_offsets;         // [T * M + 1]
  FfiInt64Vec factor_score_offsets;      // [T * F + 1]
  FfiInt32Vec policy_to_modality_idx;    // [T * M * P]
  FfiInt32Vec factor_policy_to_history;  // [T * F * P]
  FfiInt64Vec mod_history_strides;       // [T * M * kMaxFfiDependencyRank]
  FfiInt32Vec mod_history_tuples;        // optional [total_mod_entries * kMaxFfiDependencyRank]
  FfiInt32Vec mod_h_dims;                // optional [T * M * rank_dim_cap]
  int64_t     total_mod_entries    = 0;
  int64_t     total_factor_entries = 0;
  int         max_history_count    = 0;

  // CUDA scratch planning. CPU ignores these fields.
  size_t modality_tmp_qo_max_floats = 0;
  size_t split_tmp_lin_max_floats   = 0;
  size_t q01_outer_max_floats       = 0;
};

inline int max_factor_history_count(const FactorHistoryLevels& histories) {
  int max_h = 1;
  for (const FactorHistoryRow& tlevel : histories) {
    for (const FactorHistoryLevel& level : tlevel) {
      max_h = std::max(max_h, level.n_histories);
    }
  }
  return max_h;
}

inline int64_t max_factor_histories_for_factor(const FactorHistoryLevels& histories, int64_t T, int64_t f) {
  int64_t max_h = 1;
  for (int64_t t = 0; t < T; ++t) {
    max_h = std::max<int64_t>(max_h, histories[t][f].n_histories);
  }
  return max_h;
}

// Shared reset for the FactorHistoryTables fields that both build_*_tables
// functions clear at entry. `with_strides` true zero-init mod_history_strides
// to ones (used by the Cartesian path); false leaves it empty (dedup path
// computes its own indexing via mod_history_tuples).
inline void reset_factor_history_tables(const Layout& L, FactorHistoryTables* tables, bool with_strides) {
  tables->mod_score_offsets.assign(static_cast<size_t>(L.T) * L.M + 1, 0);
  tables->factor_score_offsets.assign(static_cast<size_t>(L.T) * L.F + 1, 0);
  tables->policy_to_modality_idx.assign(static_cast<size_t>(L.T) * L.M * L.P, 0);
  tables->factor_policy_to_history.resize(static_cast<size_t>(L.T) * L.F * L.P);
  if (with_strides) {
    tables->mod_history_strides.assign(static_cast<size_t>(L.T) * L.M * kMaxFfiDependencyRank, 1);
  } else {
    tables->mod_history_strides.clear();
  }
  tables->mod_history_tuples.clear();
  tables->mod_h_dims.clear();
  tables->modality_tmp_qo_max_floats = 0;
  tables->split_tmp_lin_max_floats   = 0;
  tables->q01_outer_max_floats       = 0;
}

// Shared factor-side fill: factor_score_offsets prefix sum + memcpy each
// per-(t, f) policy_to_history into factor_policy_to_history.
// Mod-side offsets / total must already be populated by the caller (the two
// build functions compute these differently).
inline void fill_factor_offsets_and_p2h(const Layout& L, const FactorHistoryLevels& histories,
                                        FactorHistoryTables* tables) {
  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t f = 0; f < L.F; ++f) {
      const int64_t idx                     = t * L.F + f;
      tables->factor_score_offsets[idx + 1] = tables->factor_score_offsets[idx] + histories[t][f].n_histories;
    }
  }
  tables->total_factor_entries = tables->factor_score_offsets[static_cast<size_t>(L.T) * L.F];
  tables->max_history_count    = max_factor_history_count(histories);
  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t f = 0; f < L.F; ++f) {
      std::memcpy(tables->factor_policy_to_history.data() + (t * L.F + f) * L.P,
                  histories[t][f].policy_to_history.data(), static_cast<size_t>(L.P) * sizeof(int32_t));
    }
  }
}

inline void build_factor_history_tables(const Layout& L, const FactorHistoryLevels& histories,
                                        FactorHistoryTables* tables, int modality_rank_dim_cap = 0, int Bn = 1) {
  reset_factor_history_tables(L, tables, /*with_strides=*/true);

  if (modality_rank_dim_cap > 0) {
    tables->mod_h_dims.assign(static_cast<size_t>(L.T) * L.M * modality_rank_dim_cap, 1);
  }

  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t m = 0; m < L.M; ++m) {
      const DependencyView deps      = modality_state_deps(L, m);
      int64_t              n_entries = 1;
      for (int i = 0; i < deps.rank; ++i) {
        const int H_i = histories[t][deps.factors[i]].n_histories;
        n_entries *= H_i;
        if (modality_rank_dim_cap > 0 && i < modality_rank_dim_cap) {
          tables->mod_h_dims[factor_history_tm_rank_dim_index(t, m, i, L.M, modality_rank_dim_cap)] = H_i;
        }
      }
      const int64_t idx                  = factor_history_tm_index(t, m, L.M);
      tables->mod_score_offsets[idx + 1] = tables->mod_score_offsets[idx] + n_entries;

      int64_t stride = 1;
      for (int i = deps.rank - 1; i >= 0; --i) {
        tables->mod_history_strides[(idx * kMaxFfiDependencyRank) + i] = stride;
        stride *= histories[t][deps.factors[i]].n_histories;
      }

      if (modality_rank_dim_cap == 3 && (deps.rank == 2 || deps.rank == 3)) {
        const int H0 = histories[t][deps.factors[0]].n_histories;
        const int H1 = histories[t][deps.factors[1]].n_histories;
        const int S0 = dep_state_size(L, deps, 0);
        const int S1 = dep_state_size(L, deps, 1);
        const int Om = static_cast<int>(L.O[m]);
        tables->q01_outer_max_floats =
            std::max(tables->q01_outer_max_floats, static_cast<size_t>(Bn) * S0 * S1 * H0 * H1);

        if (deps.rank == 2) {
          tables->modality_tmp_qo_max_floats =
              std::max(tables->modality_tmp_qo_max_floats, static_cast<size_t>(Bn) * Om * H0 * H1);
        } else {
          const int S_split = dep_state_size(L, deps, 2);
          tables->modality_tmp_qo_max_floats =
              std::max(tables->modality_tmp_qo_max_floats, static_cast<size_t>(Bn) * H0 * H1 * Om * S_split);
          tables->split_tmp_lin_max_floats =
              std::max(tables->split_tmp_lin_max_floats, static_cast<size_t>(Bn) * H0 * H1 * S_split);
        }
      }
    }
  }
  tables->total_mod_entries = tables->mod_score_offsets[static_cast<size_t>(L.T) * L.M];

  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t m = 0; m < L.M; ++m) {
      const DependencyView deps    = modality_state_deps(L, m);
      const int64_t        idx     = factor_history_tm_index(t, m, L.M);
      const int64_t*       strides = tables->mod_history_strides.data() + idx * kMaxFfiDependencyRank;
      for (int64_t p = 0; p < L.P; ++p) {
        int64_t flat = 0;
        for (int i = 0; i < deps.rank; ++i) {
          flat += static_cast<int64_t>(histories[t][deps.factors[i]].policy_to_history[p]) * strides[i];
        }
        tables->policy_to_modality_idx[(idx * L.P) + p] = static_cast<int32_t>(flat);
      }
    }
  }

  fill_factor_offsets_and_p2h(L, histories, tables);
}

inline void build_observed_factor_history_tables(const Layout& L, const FactorHistoryLevels& histories,
                                                 FactorHistoryTables* tables) {
  reset_factor_history_tables(L, tables, /*with_strides=*/false);

  std::unordered_map<uint64_t, FfiInt32Vec> buckets;
  int32_t                                   tuple[kMaxFfiDependencyRank];
  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t m = 0; m < L.M; ++m) {
      const DependencyView deps = modality_state_deps(L, m);
      const int64_t        tm   = factor_history_tm_index(t, m, L.M);
      const int64_t        base = static_cast<int64_t>(tables->mod_history_tuples.size()) / kMaxFfiDependencyRank;
      buckets.clear();

      for (int64_t p = 0; p < L.P; ++p) {
        uint64_t hash = kFactorTreeFNVBasis;
        for (int i = 0; i < deps.rank; ++i) {
          tuple[i] = histories[t][deps.factors[i]].policy_to_history[p];
          hash     = factor_tree_mix64(hash, static_cast<uint64_t>(static_cast<uint32_t>(tuple[i])));
        }

        int32_t found = -1;
        auto    it    = buckets.find(hash);
        if (it != buckets.end()) {
          for (int32_t cand : it->second) {
            const int32_t* cand_tuple = tables->mod_history_tuples.data() + (base + cand) * kMaxFfiDependencyRank;
            bool           match      = true;
            for (int i = 0; i < deps.rank; ++i) {
              if (cand_tuple[i] != tuple[i]) {
                match = false;
                break;
              }
            }
            if (match) {
              found = cand;
              break;
            }
          }
        }

        if (found < 0) {
          found = static_cast<int32_t>(
              (static_cast<int64_t>(tables->mod_history_tuples.size()) / kMaxFfiDependencyRank) - base);
          for (int i = 0; i < kMaxFfiDependencyRank; ++i) {
            tables->mod_history_tuples.push_back(i < deps.rank ? tuple[i] : 0);
          }
          buckets[hash].push_back(found);
        }
        tables->policy_to_modality_idx[(tm * L.P) + p] = found;
      }
      tables->mod_score_offsets[tm + 1] =
          static_cast<int64_t>(tables->mod_history_tuples.size()) / kMaxFfiDependencyRank;
    }
  }
  tables->total_mod_entries = tables->mod_score_offsets[static_cast<size_t>(L.T) * L.M];
  fill_factor_offsets_and_p2h(L, histories, tables);
}

}  // namespace pymdp_ffi
