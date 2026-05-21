// Per-batch driver + final scatter for the NegEfeCpu kernel. Sits between the
// hot per-level work (Kernel::run_level in neg_efe_cpu_core.h) and the ABI
// entry (NegEfeCpu in neg_efe_cpu_entry.cc).

#include "neg_efe/neg_efe_cpu_pipeline.h"

#include <algorithm>
#include <cstdint>
#include <utility>

#include <omp.h>

#include "common/error_helpers.h"
#include "common/kernel_primitives.h"
#include "neg_efe/factor_history_tables.h"
#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_cpu_core.h"
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"

namespace pymdp_ffi {
namespace {

inline void scatter_factor_history(const Layout& L, const FactorHistoryTables& tables, const float* modality_scores,
                                   const float* factor_scores, float* out_data) {
  auto scatter_one_policy = [&](int64_t p) {
    float sum = 0.0f;
    for (int64_t t = 0; t < L.T; ++t) {
      for (int64_t m = 0; m < L.M; ++m) {
        const int64_t tm  = factor_history_tm_index(t, m, L.M);
        const int64_t off = tables.mod_score_offsets[tm];
        const int32_t idx = tables.policy_to_modality_idx[(tm * L.P) + p];
        sum += modality_scores[off + idx];
      }
      for (int64_t f = 0; f < L.F; ++f) {
        const int64_t off = tables.factor_score_offsets[t * L.F + f];
        const int32_t idx = tables.factor_policy_to_history[((t * L.F + f) * L.P) + p];
        sum += factor_scores[off + idx];
      }
    }
    out_data[p] = sum;
  };

  // Scatter is L.P * L.T * (L.M + L.F) trivial float adds — for agent_step
  // (L.P=24, L.T=1, L.M+L.F=6 → 144 adds) the libgomp fork+barrier dwarfs the
  // work, so an L.P-only gate misses the regression on ARMv8.0 hosts where
  // kOmpNodeThreshold=8 lets 24 policies fire parallel. Combine an absolute
  // op-count gate with the host's L.P threshold so big inductive fixtures
  // (L.P=1728, L.T=3, L.M+L.F=6 → 31K) still parallelize.
  constexpr int64_t kNegEfeScatterOmpWorkThreshold = 4096;
  const int64_t     scatter_work                   = L.P * L.T * (L.M + L.F);
  if (scatter_work < kNegEfeScatterOmpWorkThreshold || L.P < kOmpNodeThreshold) {
    for (int64_t p = 0; p < L.P; ++p) {
      scatter_one_policy(p);
    }
    return;
  }
#pragma omp parallel for schedule(static) num_threads(omp_team_size())
  for (int64_t p = 0; p < L.P; ++p) {
    scatter_one_policy(p);
  }
}

}  // namespace

FfiError process_batch_element(const Layout& L, int32_t flags, const BatchInputs& in, CallScratch& sc) {
  // Layout invariants from validate_layout — restate so the optimizer can drop
  // zero-trip guards across this function-boundary frame.
  if (L.F <= 0 || L.M <= 0 || L.P <= 0 || L.T <= 0) __builtin_unreachable();

  // Memoize factor-history build across broadcast-vmap batch elements and
  // across rollout steps. pol is identical across the vmap batch in
  // Agent.infer_policies (broadcast_all) and constant across rollout steps
  // (policies are fixed at agent init), so the dedup work amortizes to ~O(1)
  // per kernel call after the first build.
  const int64_t  pm_size = L.P * L.T * L.F;
  const uint64_t sig     = factor_history_pm_sig(L);
  const uint64_t tag     = content_tag(reinterpret_cast<const float*>(in.pm), pm_size);
  if (PYMDP_UNLIKELY(sc.pm_size != pm_size || sc.pm_layout_sig != sig || sc.pm_content_tag != tag)) {
    PYMDP_TRY(build_factor_history_host(L, in.pm, &sc.factor_tree));
    build_observed_factor_history_tables(L, sc.factor_tree, &sc.tables);
    sc.pm_size        = pm_size;
    sc.pm_layout_sig  = sig;
    sc.pm_content_tag = tag;
  }
  sc.ensure_size(L);

  Kernel k(L, flags, in);

  QsLevel curr = initial_qs_level(L, in.qs);
  QsLevel next;
  for (int64_t t = 0; t < L.T; ++t) {
    const int slot = static_cast<int>(t & 1);
    k.run_level(t, sc.factor_tree[t], sc.tables, curr, &next, sc.qs_factor_buf[slot],
                sc.factor_scores.data() + sc.tables.factor_score_offsets[t * L.F], sc.modality_scores.data());
    std::swap(curr, next);
  }

  scatter_factor_history(L, sc.tables, sc.modality_scores.data(), sc.factor_scores.data(), in.out);
  return FfiError::Success();
}

}  // namespace pymdp_ffi
