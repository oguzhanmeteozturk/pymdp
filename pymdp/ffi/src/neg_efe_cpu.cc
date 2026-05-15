// Fused neg-EFE kernel for pymdp. Replaces the policy-wise JAX neg-EFE vmap in
// update_posterior_policies{,_inductive} and returns pre-softmax policy scores.
//
// Shape/feature limits:
//   * A_dependencies and B_dependencies arbitrary rank <= 8. Cost scales with
//     K = prod(S[deps]); factor-local B is the K_f = S_f special case.
//
// Schedule: cache A/B/wA/wB precomputes; rebuild light metadata per call; dedup
// policy prefixes into an action tree; walk levels with rolling qs buffers,
// fused A/H_A scoring, optional pA/pB novelty, and optional inductive value;
// scatter to policy vector.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include <omp.h>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "error_helpers.h"
#include "factor_history_tables.h"
#include "factor_history_tree.h"
#include "kernel_primitives.h"
#include "neg_efe.h"
#include "neg_efe_entry.h"
#include "neg_efe_precompute.h"

namespace ffi = ::xla::ffi;

namespace pymdp_ffi {
namespace {

// Per-vmap-batch-element pointers carved out of NegEfeCpu's flat buffers.
// Bundled so per-batch arithmetic happens in one place. Consumed by the Kernel
// ctor and process_batch_element. pA / pB are nullptr when use_param_info_gain
// is off. `__restrict__` on every pointer field: each maps to a distinct XLA
// input/output buffer with no possible aliasing — no-alias propagates through
// downstream dereferences in the Kernel hot path.
struct BatchInputs {
  const int32_t* __restrict__ pm;
  const float* __restrict__ qs;
  const float* __restrict__ A;
  const float* __restrict__ B;
  const float* __restrict__ C;
  const float* __restrict__ I;
  const float* __restrict__ pA;
  const float* __restrict__ pB;
  float inductive_eps;
  float* __restrict__ out;
};

// kOmpNodeThreshold (from kernel_primitives.h) gates per-level node parallelism,
// modality scoring, policy scatter, and the outer batch loop.

// =============================================================================
// Per-call types & scratch
// =============================================================================

struct QsLevel {
  std::vector<const float*> ptrs;
  std::vector<int32_t>      histories;
};

inline QsLevel initial_qs_level(const Layout& L, const float* qs_init) {
  QsLevel level;
  level.ptrs.resize(L.F);
  level.histories.assign(L.F, 1);
  for (int64_t f = 0; f < L.F; ++f) {
    level.ptrs[f] = qs_init + L.qs_off[f];
  }
  return level;
}

template <class OutT>
inline const OutT* build_qs_outer_history_typed(const Layout& L, const DependencyView& deps, const QsLevel& level,
                                                const int32_t* histories, int histories_len, float* interm,
                                                float* spare, OutT* out) {
  if (deps.rank == 0) {
    out[0] = static_cast<OutT>(1.0f);
    return out;
  }
  // Caller supplies histories_len contiguous ids (parent row or modality decode buffer).
  // Layout validation caps deps.rank <= kMaxFfiDependencyRank.
  if (histories_len < deps.rank) {
    out[0] = static_cast<OutT>(1.0f);
    return out;
  }

  std::array<int32_t, kMaxFfiDependencyRank> hloc{};
  for (int j = 0; j < deps.rank; ++j) {
    hloc[j] = histories[j];
  }

  if (deps.rank == 1) {
    const int64_t f = deps.factors[0];
    const float*  q = level.ptrs[f] + static_cast<int64_t>(hloc[0]) * L.S[f];
    if constexpr (std::is_same<OutT, float>::value) {
      return q;
    } else {
      pack_f32_to_f16(L.S[f], q, out);
      return out;
    }
  }

  const int64_t f0 = deps.factors[0];
  const int64_t f1 = deps.factors[1];
  const float*  q0 = level.ptrs[f0] + static_cast<int64_t>(hloc[0]) * L.S[f0];
  const float*  q1 = level.ptrs[f1] + static_cast<int64_t>(hloc[1]) * L.S[f1];
  if (deps.rank == 2) {
    kron_final_outer<OutT>(L.S[f0], q0, L.S[f1], q1, out);
    return out;
  }

  kron_final_outer<float>(L.S[f0], q0, L.S[f1], q1, interm);
  const float* cur   = interm;
  float*       next  = spare;
  int64_t      cur_k = L.S[f0] * L.S[f1];

  // Rank 3 is the fixed tail kron(q0,q1,q2); ranks >= 4 extend via the loop then one final kron.
  if (deps.rank == 3) {
    const int64_t f2 = deps.factors[2];
    const float*  q2 = level.ptrs[f2] + static_cast<int64_t>(hloc[2]) * L.S[f2];
    kron_final_outer<OutT>(cur_k, cur, L.S[f2], q2, out);
    return out;
  }

  for (int i = 2; i < deps.rank - 1; ++i) {
    const int64_t f = deps.factors[i];
    const float*  q = level.ptrs[f] + static_cast<int64_t>(hloc[i]) * L.S[f];
    kron_final_outer<float>(cur_k, cur, L.S[f], q, next);
    cur_k *= L.S[f];
    cur  = next;
    next = (next == spare) ? interm : spare;
  }
  // rank >= 4 here: rank in {0,1,2,3} all returned above; rank <= kMaxFfiDependencyRank validated
  // at layout-build time. The unreachable hint silences a static-analyzer warning about a
  // speculative negative tail_i. `if (!x) __builtin_unreachable()` is the portable form;
  // clang's __builtin_assume isn't available on gcc.
  if (deps.rank < 4 || deps.rank > kMaxFfiDependencyRank) __builtin_unreachable();
  const int     tail_i = deps.rank - 1;
  const int64_t f_tail = deps.factors[tail_i];
  const float*  q_tail = level.ptrs[f_tail] + static_cast<int64_t>(hloc[tail_i]) * L.S[f_tail];
  kron_final_outer<OutT>(cur_k, cur, L.S[f_tail], q_tail, out);
  return out;
}

inline const float* build_qs_outer_history(const Layout& L, const DependencyView& deps, const QsLevel& level,
                                           const int32_t* histories, int histories_len, float* interm, float* spare,
                                           float* out) {
  return build_qs_outer_history_typed<float>(L, deps, level, histories, histories_len, interm, spare, out);
}

inline void decode_history_tuple(int64_t idx, const int64_t* strides, int rank, int32_t* histories) {
  int64_t rem = idx;
  for (int i = 0; i < rank; ++i) {
    histories[i] = static_cast<int32_t>(rem / strides[i]);
    rem -= static_cast<int64_t>(histories[i]) * strides[i];
  }
}

// Per-thread scratch for the inner per-node work. `qo_plus_HA` is sized
// max_O + 1 so a single sgemv can write the optional qs_HA row at position
// O_m. `interm_f32` / `interm_spare` host the Kronecker product's f32
// intermediates for dep_rank >= 2 (shared across A modalities and B factors).
// `qs_outer_b_f32` is the B-propagation Kronecker output buffer; rank 0 writes
// a scalar there, rank 1 aliases qs[deps[0]], rank >= 2 writes the full
// Kronecker product.
struct NodeScratch {
  std::vector<float>  qo_plus_HA;
  std::vector<float>  interm_f32;
  std::vector<float>  interm_spare;
  std::vector<FfiF16> qs_outer_compact;
  std::vector<float>  qs_outer_b_f32;
  // Param-info-gain scratch — only sized when use_param_info_gain is on.
  // qs_outer_a_f32[max_K]: f32 qs_outer for wA pass (rebuilt per modality;
  //   the standard A path uses a compact FfiF16 qs_outer that f16/FML sgemv
  //   consumes, but wA is f32 so we need an f32 view too).
  // wa_qs[O_m]: (wA[m] @ qs_outer) target, dotted with qo_plus_HA's qo prefix.
  // wb_qs[S_f]: (wB[f, u] @ qs_outer_b) target, dotted with qs_next[f].
  std::vector<float> qs_outer_a_f32;
  std::vector<float> wa_qs;
  std::vector<float> wb_qs;

  // Resize-up-only: pymdp call patterns repeat shapes across calls; retaining
  // capacity hits zero allocations in steady state.
  void ensure_size(int64_t max_O, int64_t max_K, int64_t max_inner_K, int64_t max_K_b, int64_t max_S) {
    ensure_at_least(qo_plus_HA, max_O + 1);
    ensure_at_least(interm_f32, std::max<int64_t>(max_inner_K, 1));
    ensure_at_least(interm_spare, std::max<int64_t>(max_inner_K, 1));
    ensure_at_least(qs_outer_compact, max_K);
    ensure_at_least(qs_outer_b_f32, std::max<int64_t>(max_K_b, 1));
    ensure_at_least(qs_outer_a_f32, std::max<int64_t>(max_K, 1));
    ensure_at_least(wa_qs, std::max<int64_t>(max_O, 1));
    ensure_at_least(wb_qs, std::max<int64_t>(max_S, 1));
  }
};

inline thread_local NodeScratch g_node_scratch{};

// Per-call scratch: factor-history trees/tables, rolling per-factor qs buffers,
// and score tables consumed by final scatter. Lives outside any OMP region —
// single thread_local slot is race-free. Resize-up-only.
//
// `pm_content_tag` / `pm_layout_sig` memoize the most-recent factor-history
// build keyed on the policy-matrix content sample + B-dep layout. The pol
// matrix is broadcast across the vmap batch in Agent.infer_policies (4
// identical buffers per call) and constant across rollout steps (policies are
// fixed at agent init), so the first batch element pays the dedup cost and
// every subsequent build_factor_history_host / build_observed_factor_history_tables
// short-circuits — matching the same broadcast-vmap caching pattern that
// g_a_cache_lru / g_b_cache_lru use for A and B.
struct CallScratch {
  FactorHistoryLevels factor_tree;
  FactorHistoryTables tables;
  PerFactorFloatRows  qs_factor_buf[2];  // [slot][F][H_f_max * S_f]
  std::vector<float>  modality_scores;   // [total_mod_entries]
  std::vector<float>  factor_scores;     // [total_factor_entries]
  uint64_t            pm_content_tag = 0;
  uint64_t            pm_layout_sig  = 0;
  int64_t             pm_size        = -1;  // -1 sentinel: cache empty.

  void ensure_size(const Layout& L) {
    for (auto& slot : qs_factor_buf) {
      slot.resize(L.F);
    }
    for (int64_t f = 0; f < L.F; ++f) {
      const int64_t max_h = max_factor_histories_for_factor(factor_tree, L.T, f);
      const size_t  n     = static_cast<size_t>(max_h * L.S[f]);
      for (auto& slot : qs_factor_buf) {
        ensure_at_least(slot[f], n);
      }
    }
    ensure_at_least(modality_scores, tables.total_mod_entries);
    ensure_at_least(factor_scores, tables.total_factor_entries);
  }
};
inline thread_local CallScratch g_call_scratch{};

// =============================================================================
// Kernel
// =============================================================================

// Owns once-per-call state shared by the level walk and modality scoring.
//
// HA, A_aug_compact, and B_tr point into thread-local caches. Metadata is
// rebuilt per call because its dependency pointers reference this call's
// Layout spans.
struct Kernel {
  const Layout& L;
  KernelFlags   flags;

  MetaSummary metas;

  const TransposedB*   B_tr;           // -> selected B-cache entry
  const PrecomputedHA* HA;             // -> selected A-cache entry
  PrecomputedInductive inductive;      // owned (varies per call: depends on qs_init)
  const FfiF16*        A_aug_compact;  // -> selected A-cache entry

  const float* A_data;
  const float* C_data;

  // Param-info-gain weights. Backed by per-thread LRU caches (g_wa_cache_lru /
  // g_wb_cache_lru) — pA / pB are usually broadcast across the vmap batch, so
  // the first element pays the precompute and the rest hit the cache. Pointers
  // are nullptr when use_param_info_gain is off or the corresponding pA / pB
  // input was missing. wA layout matches A (`L.A_off`-indexed); wB layout
  // matches the transposed B (B_tr.offsets-indexed) so the per-factor B-rollout
  // sgemv pattern reuses the same offset math.
  const float*       wA_data = nullptr;
  const TransposedB* wB_tr   = nullptr;

  // Order matters: a-cache before b-cache before metas (build_meta_summary
  // reads *HA and *B_tr); inductive and param-info-gain precomputes are
  // independent.
  Kernel(const Layout& L_, int32_t flag_bits, const BatchInputs& in)
      : L(L_), flags(KernelFlags::from_bits(flag_bits)), A_data(in.A), C_data(in.C) {
    ACache* a     = refresh_cache_lru(g_a_cache_lru, in.A, L.A_off[L.M], a_layout_sig(L), [&](ACache& c) {
      c.HA                      = precompute_HA(L, in.A);
      ModalityMetaVec tmp_metas = build_modality_metas(L, c.HA);
      c.A_aug_compact           = build_A_aug_compact(L, tmp_metas, in.A, c.HA);
    });
    HA            = &a->HA;
    A_aug_compact = a->A_aug_compact.data();

    BCache* b = refresh_cache_lru(g_b_cache_lru, in.B, L.B_off[L.F], b_layout_sig(L),
                                  [&](BCache& c) { c.Btr = transpose_B(L, in.B); });
    B_tr      = &b->Btr;

    if (flags.use_inductive) inductive = precompute_inductive(L, in.qs, in.I, in.inductive_eps);
    if (flags.use_param_info_gain) {
      // Cache wA / wB across vmap batch elements (and across calls of the same
      // shape). pA / pB are typically broadcast across batch in
      // Agent.infer_policies — without this LRU every batch element redoes the
      // digamma+log precompute. Cache key reuses a_layout_sig / b_layout_sig
      // because wA / wB layouts are determined entirely by A / B layouts.
      if (in.pA != nullptr) {
        WACache* w = refresh_cache_lru(g_wa_cache_lru, in.pA, L.A_off[L.M], a_layout_sig(L),
                                       [&](WACache& c) { c.wA = precompute_wA(L, in.pA); });
        wA_data    = w->wA.data();
      }
      if (in.pB != nullptr) {
        WBCache* w = refresh_cache_lru(g_wb_cache_lru, in.pB, L.B_off[L.F], b_layout_sig(L),
                                       [&](WBCache& c) { c.wBtr = precompute_wB_transposed(L, in.pB); });
        wB_tr      = &w->wBtr;
      }
    }
    metas = build_meta_summary(L, *HA, *B_tr);
  }

  // Score one modality-history tuple.
  //   * `qs_outer_compact` is the modality dependency Kronecker product packed
  //     into FfiF16 (on ARMv8.2/FML hosts; on ARMv8.0 hosts FfiF16 == float so
  //     it can alias an f32 buffer).
  //   * `qs_outer_f32_or_null` is the same product in f32. Required when
  //     pA novelty is live (wA is f32 only — digamma/log precompute). Null
  //     otherwise; pA novelty silently skips.
  //
  // The A pass uses qs_outer_compact (FML fast path or v8.0 f32 reinterpret);
  // pA novelty uses qs_outer_f32_or_null. Rank-0 modalities skip the qs_outer
  // dependency entirely (qo collapses to A_m).
  float score_modality(const ModalityMeta& mm, int64_t t, const FfiF16* qs_outer_compact,
                       const float* qs_outer_f32_or_null, NodeScratch& s) const {
    float delta = 0.0f;
    if (mm.state_deps.rank == 0) {
      std::memcpy(s.qo_plus_HA.data(), A_data + mm.offsets.A_input, sizeof(float) * mm.O);
      if (flags.use_states_info_gain) {
        delta += entropy(s.qo_plus_HA.data(), mm.O) - HA->data[mm.offsets.entropy];
      }
    } else {
#if PYMDP_FFI_HAS_F16_FML
      // ARMv8.2 FP16/FML fast path: one fused f16 sgemv produces qo
      // (rows 0..O_m-1) and the optional qs_HA scalar (row O_m).
      const int64_t rows = mm.O + (flags.use_states_info_gain ? 1 : 0);
      sgemv_rm_compact(rows, mm.K, A_aug_compact + mm.offsets.A_aug, mm.K, qs_outer_compact, s.qo_plus_HA.data());
      if (flags.use_states_info_gain) {
        delta += entropy(s.qo_plus_HA.data(), mm.O) - s.qo_plus_HA[mm.O];
      }
#else
      // ARMv8.0 fallback (Cortex-A57): no FP16/FML, no packed
      // A_aug. Score directly against the f32 source A then reduce HA · qs
      // separately. Same FMA count as FML; the win is avoiding an A_aug
      // copy that pressures Cortex-A57's 2 MB L2.
      static_assert(std::is_same<FfiF16, float>::value, "v8.0 path requires f32 storage");
      const float* qs_outer_f32 = reinterpret_cast<const float*>(qs_outer_compact);
      sgemv_rm_f32(mm.O, mm.K, A_data + mm.offsets.A_input, mm.K, qs_outer_f32, s.qo_plus_HA.data());
      if (flags.use_states_info_gain) {
        const float ha_dot = sdot_f32(mm.K, HA->data.data() + mm.offsets.entropy, qs_outer_f32);
        delta += entropy(s.qo_plus_HA.data(), mm.O) - ha_dot;
      }
#endif
    }

    if (flags.use_utility) {
      const float* Cm_t = C_data + mm.offsets.C_input + t * mm.O;
      delta += sdot_f32(mm.O, s.qo_plus_HA.data(), Cm_t);
    }

    // pA novelty contribution: pA_term = qo · (wA[m] @ qs_outer). Mirrors
    // calc_negative_pA_info_gain in pymdp.control; sign-net positive in the
    // policy score after JAX's negation conventions. Skipped for rank-0
    // modalities (no state-dependent novelty term).
    if (flags.use_param_info_gain && wA_data != nullptr && mm.state_deps.rank > 0 && qs_outer_f32_or_null != nullptr) {
      sgemv_rm_f32(mm.O, mm.K, wA_data + mm.offsets.A_input, mm.K, qs_outer_f32_or_null, s.wa_qs.data());
      delta += sdot_f32(mm.O, s.qo_plus_HA.data(), s.wa_qs.data());
    }
    return delta;
  }

  // Process every history at one tree level: factor rollout (B-propagation +
  // optional pB-novelty + optional inductive) followed by modality scoring.
  // Single OMP fork per level — workers fan out across factor histories
  // (Phase 1) and modality entries (Phase 2) within one parallel region. An
  // explicit barrier separates the phases so qs_next is fully written before
  // modality scoring reads it. `nowait` between sibling factors / modalities
  // is safe because each writes a disjoint slice (next_storage[f] /
  // factor_scores_f / score_m are all factor- or modality-keyed).
  void run_level(int64_t t, const FactorHistoryRow& factors_level, const FactorHistoryTables& tables,
                 const QsLevel& curr, QsLevel* next, PerFactorFloatRows& next_storage, float* factor_score_base,
                 float* modality_score_base) const {
    // Master sets next->ptrs/histories once before the parallel region; all
    // workers read them in Phase 2 via shared memory (the implicit barrier on
    // each phase-1 omp for + the explicit phase barrier guarantees visibility).
    next->ptrs.resize(L.F);
    next->histories.resize(L.F);
    for (int64_t f = 0; f < L.F; ++f) {
      next->ptrs[f]      = next_storage[f].data();
      next->histories[f] = factors_level[f].n_histories;
    }

    // Decide whether to fire OMP. Two gates that both have to pass:
    //   * level_flops >= kNegEfeLevelOmpFlopThreshold — total work has to
    //     amortize the libomp fork+barrier (~50K FMAs at M3 Max NEON
    //     throughput ÷ 12 perf cores). Without this, agent_step's tiny
    //     [8,6,4] state shape (~40K FMAs/level) burns more on fork than
    //     it saves.
    //   * max_subloop_iter >= kOmpNodeThreshold — the largest single
    //     `omp for` inside the parallel region has to distribute usefully
    //     across the team. rollout_loop hits this case: level_flops is
    //     ~1.6M but every sub-loop has ~12 items (P=12 with factor-local
    //     layout), so a 12-thread team gives 1 item/thread + barrier-wait
    //     and net-regresses vs serial. Inductive / param_info_gain leaf
    //     levels have 200+ entries per modality and clear this gate.
    int64_t level_flops      = 0;
    int64_t max_subloop_iter = 0;
    for (int64_t f = 0; f < L.F; ++f) {
      const FactorMeta& fm = metas.factors[f];
      const int64_t     n  = factors_level[f].n_histories;
      level_flops += n * fm.S * fm.K;
      max_subloop_iter = std::max(max_subloop_iter, n);
    }
    const int64_t mod_extra_row = flags.use_states_info_gain ? 1 : 0;
    for (int64_t m = 0; m < L.M; ++m) {
      const ModalityMeta& mm      = metas.modalities[m];
      const int64_t       tm      = factor_history_tm_index(t, m, L.M);
      const int64_t       local_n = tables.mod_score_offsets[tm + 1] - tables.mod_score_offsets[tm];
      level_flops += local_n * (mm.O + mod_extra_row) * mm.K;
      max_subloop_iter = std::max(max_subloop_iter, local_n);
    }
    const bool use_omp = level_flops >= kNegEfeLevelOmpFlopThreshold && max_subloop_iter >= kOmpNodeThreshold;

    auto run_factor_phase = [&](NodeScratch& s, bool use_omp) {
      int64_t f_score_off = 0;
      for (int64_t f = 0; f < L.F; ++f) {
        const FactorHistoryLevel& lv              = factors_level[f];
        const FactorMeta&         fm              = metas.factors[f];
        float*                    qs_out          = next_storage[f].data();
        float*                    factor_scores_f = factor_score_base + f_score_off;
        f_score_off += lv.n_histories;
        // `nowait` lets workers race ahead to the next factor — safe because
        // (qs_out, factor_scores_f) are factor-disjoint and Phase 1 reads
        // only `curr` (filled by the previous level). The explicit barrier
        // after the factor phase fences before Phase 2 starts.
        if (use_omp) {
#pragma omp for schedule(static) nowait
          for (int32_t h = 0; h < lv.n_histories; ++h) {
            score_factor_history(lv, fm, f, h, curr, qs_out, factor_scores_f, s);
          }
        } else {
          for (int32_t h = 0; h < lv.n_histories; ++h) {
            score_factor_history(lv, fm, f, h, curr, qs_out, factor_scores_f, s);
          }
        }
      }
    };

    auto run_modality_phase = [&](NodeScratch& s, bool use_omp) {
      for (int64_t m = 0; m < L.M; ++m) {
        const ModalityMeta&  mm         = metas.modalities[m];
        const DependencyView deps       = mm.state_deps;
        const int64_t        tm         = factor_history_tm_index(t, m, L.M);
        const int64_t        local_n    = tables.mod_score_offsets[tm + 1] - tables.mod_score_offsets[tm];
        float*               score_m    = modality_score_base + tables.mod_score_offsets[tm];
        const int64_t*       strides    = tables.mod_history_strides.empty()
                                              ? nullptr
                                              : tables.mod_history_strides.data() + tm * kMaxFfiDependencyRank;
        const int64_t        tuple_base = tables.mod_score_offsets[tm] * kMaxFfiDependencyRank;
        if (use_omp) {
#pragma omp for schedule(static) nowait
          for (int64_t idx = 0; idx < local_n; ++idx) {
            score_modality_entry(t, mm, deps, *next, tables, idx, tuple_base, strides, score_m, s);
          }
        } else {
          for (int64_t idx = 0; idx < local_n; ++idx) {
            score_modality_entry(t, mm, deps, *next, tables, idx, tuple_base, strides, score_m, s);
          }
        }
      }
    };

    if (!use_omp) {
      // Serial path: skip libomp entirely on small per-level work or on
      // levels where no sub-loop has enough iterations to distribute.
      NodeScratch& s = g_node_scratch;
      s.ensure_size(metas.max_O, metas.max_K, metas.max_inner_K_any, metas.max_K_b, metas.max_S);
      run_factor_phase(s, /*use_omp=*/false);
      run_modality_phase(s, /*use_omp=*/false);
      return;
    }

#pragma omp parallel num_threads(omp_team_size())
    {
      NodeScratch& s = g_node_scratch;
      s.ensure_size(metas.max_O, metas.max_K, metas.max_inner_K_any, metas.max_K_b, metas.max_S);
      run_factor_phase(s, /*use_omp=*/true);
#pragma omp barrier
      run_modality_phase(s, /*use_omp=*/true);
    }
  }

  // Phase-1 body: rollout factor f's history h from `curr` through B[f, u_h]
  // into qs_out + h*S_f, then accumulate optional pB-novelty + inductive into
  // factor_scores_f[h]. Pure compute; no OMP / cross-thread state.
  void score_factor_history(const FactorHistoryLevel& lv, const FactorMeta& fm, int64_t f, int32_t h,
                            const QsLevel& curr, float* qs_out, float* factor_scores_f, NodeScratch& s) const {
    const int32_t* parent_h = lv.parent_histories.data() + static_cast<int64_t>(h) * lv.n_parents;
    const int32_t  u        = lv.action_per_history[h];
    const float* qs_outer = build_qs_outer_history(L, fm.transition_deps, curr, parent_h, lv.n_parents,
                                                   s.interm_f32.data(), s.interm_spare.data(), s.qs_outer_b_f32.data());
    float*       qs_next_f = qs_out + static_cast<int64_t>(h) * fm.S;
    const float* Bfu       = B_tr->data.data() + fm.offsets.B_transposed + static_cast<int64_t>(u) * fm.S * fm.K;
    sgemv_rm_f32(fm.S, fm.K, Bfu, fm.K, qs_outer, qs_next_f);

    float score = 0.0f;
    if (flags.use_param_info_gain && wB_tr != nullptr) {
      const float* wBfu = wB_tr->data.data() + fm.offsets.B_transposed + static_cast<int64_t>(u) * fm.S * fm.K;
      sgemv_rm_f32(fm.S, fm.K, wBfu, fm.K, qs_outer, s.wb_qs.data());
      score += sdot_f32(fm.S, qs_next_f, s.wb_qs.data());
    }
    if (flags.use_inductive) {
      score += sdot_f32(fm.S, inductive.data.data() + L.qs_off[f], qs_next_f);
    }
    factor_scores_f[h] = score;
  }

  // Phase-2 body: score one (m, idx) modality history tuple. When pA novelty
  // is live, we build the f32 Kronecker product and pack a compact view of it
  // for the A pass; otherwise we go straight to the compact build (saves the
  // f32 buffer trip).
  void score_modality_entry(int64_t t, const ModalityMeta& mm, const DependencyView& deps, const QsLevel& qs_next,
                            const FactorHistoryTables& tables, int64_t idx, int64_t tuple_base, const int64_t* strides,
                            float* score_m, NodeScratch& s) const {
    std::array<int32_t, kMaxFfiDependencyRank> histories{};
    if (!tables.mod_history_tuples.empty()) {
      const int32_t* tuple = tables.mod_history_tuples.data() + tuple_base + idx * kMaxFfiDependencyRank;
      for (int i = 0; i < deps.rank; ++i) histories[i] = tuple[i];
    } else {
      decode_history_tuple(idx, strides, deps.rank, histories.data());
    }
    const bool pa_path = flags.use_param_info_gain && wA_data != nullptr && deps.rank > 0;
    if (pa_path) {
      const float* qs_outer_f32 =
          build_qs_outer_history(L, deps, qs_next, histories.data(), kMaxFfiDependencyRank, s.interm_f32.data(),
                                 s.interm_spare.data(), s.qs_outer_a_f32.data());
#if PYMDP_FFI_HAS_F16_FML
      for (int64_t k = 0; k < mm.K; ++k) s.qs_outer_compact[k] = static_cast<FfiF16>(qs_outer_f32[k]);
      const FfiF16* qs_outer_compact = s.qs_outer_compact.data();
#else
      static_assert(std::is_same<FfiF16, float>::value, "v8.0 path requires f32 storage");
      const FfiF16* qs_outer_compact = reinterpret_cast<const FfiF16*>(qs_outer_f32);
#endif
      score_m[idx] = score_modality(mm, t, qs_outer_compact, qs_outer_f32, s);
    } else {
      const FfiF16* qs_outer_compact =
          build_qs_outer_history_typed<FfiF16>(L, deps, qs_next, histories.data(), kMaxFfiDependencyRank,
                                               s.interm_f32.data(), s.interm_spare.data(), s.qs_outer_compact.data());
      score_m[idx] = score_modality(mm, t, qs_outer_compact, nullptr, s);
    }
  }
};

// =============================================================================
// Per-batch driver
// =============================================================================

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

// One vmap-batch element: build factor histories, walk levels, scatter to out.
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

}  // namespace

// =============================================================================
// ABI entry point
// =============================================================================

FfiError NegEfeCpu(FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B, FfiF32Buf C, FfiF32Buf I,
                   FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon, FfiF32Out out, FfiInt64Span S_span,
                   FfiInt64Span O_span, FfiInt64Span U_span, FfiInt64Span qs_off_span, FfiInt64Span A_off_span,
                   FfiInt64Span B_off_span, FfiInt64Span C_off_span, FfiInt64Span I_off_span,
                   FfiInt64Span I_depths_span, FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span,
                   FfiInt64Span B_dep_flat_span, FfiInt64Span B_dep_off_span, int32_t flags) {
  // vmap_method="broadcast_all" prepends a leading batch dim to every buffer.
  // Detect it from policy_matrix rank and iterate per instance inside one FFI
  // dispatch; A/B cache entries are shared across broadcast batch elements.
  const LayoutSpans spans{S_span,         O_span,          U_span,        qs_off_span,   A_off_span,
                          B_off_span,     C_off_span,      I_off_span,    I_depths_span, A_dep_flat_span,
                          A_dep_off_span, B_dep_flat_span, B_dep_off_span};
  ParsedCall        pc;
  PYMDP_TRY(
      parse_and_validate_call(policy_matrix, qs_init, A, B, C, I, pA, pB, inductive_epsilon, out, spans, flags, &pc));

  // Layout invariants from validate_layout / parse_call_shape — restate so the
  // optimizer can drop zero-trip guards across the per-batch loop and downstream
  // process_batch_element / Kernel inlining boundary.
  if (pc.L.F <= 0 || pc.L.M <= 0 || pc.L.P <= 0 || pc.L.T <= 0 || pc.shape.Bn <= 0) __builtin_unreachable();

  const int32_t* pm_base  = policy_matrix.typed_data();
  const float*   qs_base  = qs_init.typed_data();
  const float*   A_base   = A.typed_data();
  const float*   B_base   = B.typed_data();
  const float*   C_base   = C.typed_data();
  const float*   I_base   = I.typed_data();
  const float*   eps_base = inductive_epsilon.typed_data();
  float*         out_base = out->typed_data();

  const float* pA_base = pA.typed_data();
  const float* pB_base = pB.typed_data();

  KernelTimer timer([&](double us) {
    std::fprintf(stderr, "[efe] Bn=%lld P=%lld T=%lld F=%lld M=%lld | total=%6.2f us\n", (long long)pc.shape.Bn,
                 (long long)pc.L.P, (long long)pc.L.T, (long long)pc.L.F, (long long)pc.L.M, us);
  });

  // Hoist all epsilon validation out of the parallel region. Cheap (Bn small),
  // and lets workers run pure compute without per-batch error returns.
  if (pc.flags.use_inductive) {
    PYMDP_TRY(validate_inductive_epsilons_batched(eps_base, pc.shape, pc.epsilon));
  }

  // Batch parallelism: each worker accesses its own thread_local CallScratch
  // and A/B/wA/wB caches via the Kernel ctor. Cache keys are content-addressed
  // so the first batch element on each worker is a miss; subsequent calls of
  // the same shape hit.
  //
  // Gate on kOmpNodeThreshold (32). Production Bn=4 stays serial so the
  // per-level OMP regions inside run_level can fan out across every core on
  // heavy leaf-level scoring (~1728 entries). With nested OMP off in
  // libgomp/libomp, firing the outer batch loop for Bn<cores collapses each
  // batch element to a single core and serializes that leaf-level work —
  // a large regression on the Bn=4 inductive fixture.
  //
  // Explicit serial/parallel split (not `omp parallel if(...)`) so the small-
  // batch path never enters the OMP runtime: an `if(false)` region still
  // creates a 1-thread team and pays libomp bookkeeping.
  FfiError   first_error      = FfiError::Success();
  const bool parallel_batches = pc.shape.Bn >= kOmpNodeThreshold;

  // pA / pB pointers stay nullptr unless the kernel was invoked with
  // use_param_info_gain AND the corresponding buffer is non-empty; the
  // Kernel ctor branches on nullptr to skip wA / wB precompute work.
  auto make_batch_inputs = [&](int64_t b) -> BatchInputs {
    return {pm_base + b * pc.strides.pm,
            qs_base + b * pc.strides.qs,
            A_base + b * pc.strides.A,
            B_base + b * pc.strides.B,
            C_base + b * pc.strides.C,
            I_base + b * pc.strides.I,
            (pc.flags.use_param_info_gain && pc.pA_present) ? pA_base + b * pc.strides.A : nullptr,
            (pc.flags.use_param_info_gain && pc.pB_present) ? pB_base + b * pc.strides.B : nullptr,
            eps_base[pc.epsilon.batched ? b : 0],
            out_base + b * pc.strides.out};
  };

  if (!parallel_batches) {
    CallScratch& sc_thread = g_call_scratch;
    for (int64_t b = 0; b < pc.shape.Bn; ++b) {
      FfiError err = process_batch_element(pc.L, flags, make_batch_inputs(b), sc_thread);
      if (err.failure() && !first_error.failure()) first_error = err;
    }
  } else {
#pragma omp parallel num_threads(static_cast<int>(std::min<int64_t>(pc.shape.Bn, omp_team_size())))
    {
      CallScratch& sc_thread = g_call_scratch;
#pragma omp for schedule(static)
      for (int64_t b = 0; b < pc.shape.Bn; ++b) {
        FfiError err = process_batch_element(pc.L, flags, make_batch_inputs(b), sc_thread);
        if (err.failure()) {
#pragma omp critical
          {
            if (!first_error.failure()) first_error = err;
          }
        }
      }
    }
  }
  if (first_error.failure()) return first_error;
  return FfiError::Success();
}

}  // namespace pymdp_ffi
