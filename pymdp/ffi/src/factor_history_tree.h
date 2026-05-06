// Per-factor action-history dedup for neg-EFE kernels.
//
// For each (t, f), a history is identified by the tuple of parent histories
// required by B_dependencies[f] at t-1 plus the factor-local action u_f at t.
// Shared by the CPU and CUDA neg-EFE backends.

#pragma once

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "error_helpers.h"
#include "neg_efe_layout.h"

namespace pymdp_ffi {

inline constexpr uint64_t kFactorTreeFNVBasis = 14695981039346656037ull;

inline uint64_t factor_tree_mix64(uint64_t h, uint64_t x) {
  h ^= x;
  h *= 1099511628211ull;
  return h;
}

struct FactorHistoryLevel {
  int n_histories = 0;
  int n_parents   = 0;  // = b_dep_rank(L, f); fixed per factor
  // [n_histories * n_parents] row-major; parent_histories[h * n_parents + i]
  // is the history id at level t-1 for parent factor B_deps[f][i]. At t=0
  // every entry is 0 (qs_init has implicit H=1).
  std::vector<int32_t> parent_histories;
  std::vector<int32_t> action_per_history;  // [n_histories]
  std::vector<int32_t> policy_to_history;   // [P]
};

// Host-side lattice built by build_factor_history_host:
// outer index = timestep t, inner = factor f (`FactorHistoryRow` spans all F).
using FactorHistoryRow    = std::vector<FactorHistoryLevel>;
using FactorHistoryLevels = std::vector<FactorHistoryRow>;

inline FfiError validate_factor_history_b_dep_ranks(const Layout& L) {
  // Redundant when L came from parse_neg_efe_layout (validate_layout already
  // checks each B_dep rank). Kept for build_factor_history_host and any caller
  // that constructs a Layout without that path.
  for (int64_t f = 0; f < L.F; ++f) {
    const DependencyView deps = factor_transition_deps(L, f);
    if (deps.rank < 1 || deps.rank > kMaxFfiDependencyRank) {
      return invalid_arg(kEfeKernelName, "B_dependencies[f] rank must be in [1, 8]");
    }
  }
  return FfiError::Success();
}

inline FfiError build_factor_history_host(const Layout& L, const int32_t* pol, FactorHistoryLevels* factor_tree_out) {
  PYMDP_TRY(validate_factor_history_b_dep_ranks(L));
  // Output shape: [T][F].
  FactorHistoryLevels& ftree = *factor_tree_out;
  ftree.assign(L.T, FactorHistoryRow(L.F));

  // FNV-1a hash over (parent histories + action); collision-checked via candidate list.
  std::unordered_map<uint64_t, std::vector<int32_t>> bucket;  // hash -> candidate history ids
  for (int64_t t = 0; t < L.T; ++t) {
    for (int64_t f = 0; f < L.F; ++f) {
      const DependencyView deps = factor_transition_deps(L, f);
      FactorHistoryLevel&  lv   = ftree[t][f];
      lv.n_histories            = 0;
      lv.n_parents              = deps.rank;
      lv.policy_to_history.assign(L.P, -1);
      bucket.clear();

      int32_t key_parents[kMaxFfiDependencyRank];
      for (int64_t p = 0; p < L.P; ++p) {
        const int32_t u = pol[(p * L.T + t) * L.F + f];
        if (u < 0 || u >= L.U[f]) {
          return invalid_arg(kEfeKernelName, "policy action out of bounds");
        }

        for (int i = 0; i < deps.rank; ++i) {
          const int64_t pf = deps.factors[i];
          key_parents[i]   = (t == 0) ? 0 : ftree[t - 1][pf].policy_to_history[p];
        }

        uint64_t hash = kFactorTreeFNVBasis;
        for (int i = 0; i < deps.rank; ++i) {
          hash = factor_tree_mix64(hash, static_cast<uint64_t>(static_cast<uint32_t>(key_parents[i])));
        }
        hash = factor_tree_mix64(hash, static_cast<uint64_t>(static_cast<uint32_t>(u)));

        int32_t found = -1;
        auto    it    = bucket.find(hash);
        if (it != bucket.end()) {
          for (int32_t cand : it->second) {
            if (lv.action_per_history[cand] != u) continue;
            bool match = true;
            for (int i = 0; i < deps.rank; ++i) {
              if (lv.parent_histories[cand * deps.rank + i] != key_parents[i]) {
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
          const int32_t hist_id = static_cast<int32_t>(lv.n_histories++);
          for (int i = 0; i < deps.rank; ++i) {
            lv.parent_histories.push_back(key_parents[i]);
          }
          lv.action_per_history.push_back(u);
          bucket[hash].push_back(hist_id);
          lv.policy_to_history[p] = hist_id;
        } else {
          lv.policy_to_history[p] = found;
        }
      }
    }
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi
