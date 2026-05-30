// Hot CPU-side surface shared across the neg-EFE TUs: per-call types
// (BatchInputs, QsLevel, NodeScratch, CallScratch), inline helpers
// (initial_qs_level, build_qs_outer_history*, decode_history_tuple), and the
// header-inline Kernel that owns score_modality + per-level walk.
//
// Co-locating these in a single header is load-bearing: the wA / wB cache
// `reinterpret_cast` alias of `qs_outer_compact` on `FfiF16 == float` hosts
// only works while `score_modality` and `build_qs_outer_*` share a translation
// unit. Inline-in-header keeps the optimizer's view of the hot path intact
// across the pipeline / entry split.

#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include <omp.h>

#include "common/error_helpers.h"
#include "common/kernel_primitives.h"
#include "neg_efe/factor_history_tables.h"
#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"

namespace pymdp_ffi {

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
  // speculative negative tail_i.
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

  // Shared-contraction scratch for the rank>=2 modality fast path
  // (score_modality_shared). `mod_T_shared` holds the per-inner-history partial
  // contraction T[h][(O + has_HA) rows][Krem] (Krem = K / S_inner);
  // `mod_qin_compact` is the f16-packed inner-factor marginal consumed by the
  // f16/FML stage-1 sgemv; `mod_qrem_f32` holds the Kronecker product of the
  // outer (first R-1) factors built per combination in stage 2. Sized on demand
  // inside the routine (resize-up-only).
  std::vector<float>  mod_T_shared;
  std::vector<FfiF16> mod_qin_compact;
  std::vector<float>  mod_qrem_f32;

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

  // Decide whether to fire OMP for one tree level. Two gates that both have to
  // pass:
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
  // Pure integer bookkeeping, runs once per level (cold relative to the
  // per-history scoring loops); inlined at -O3.
  bool decide_run_level_omp(int64_t t, const FactorHistoryRow& factors_level, const FactorHistoryTables& tables) const {
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
    return level_flops >= kNegEfeLevelOmpFlopThreshold && max_subloop_iter >= kOmpNodeThreshold;
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

    const bool use_omp = decide_run_level_omp(t, factors_level, tables);

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
          // Serial scoring: rank>=2 non-pA modalities take the shared-contraction
          // fast path (build A.q_inner once per inner history, reuse across the
          // outer factors' histories). pA novelty and rank < 2 keep the
          // per-entry routine.
          const bool pa_active = flags.use_param_info_gain && wA_data != nullptr;
          if (deps.rank >= 2 && !pa_active) {
            score_modality_shared(t, mm, *next, tables, tuple_base, strides, local_n, score_m, s);
          } else {
            for (int64_t idx = 0; idx < local_n; ++idx) {
              score_modality_entry(t, mm, deps, *next, tables, idx, tuple_base, strides, score_m, s);
            }
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

  // Shared-contraction fast path for rank>=2 modalities (serial scoring only).
  //
  // The per-entry path (score_modality_entry -> score_modality) materializes the
  // full K = prod(S_dep) Kronecker product of all dependency marginals for every
  // history combination and contracts A against it — recomputing the
  // A . q_inner contraction redundantly for every combination of the outer
  // factors' histories. JAX's factor_dot / opt_einsum shares that contraction.
  // This mirrors it with one level of sharing on the innermost dependency factor
  // (A's contiguous axis): contract A (and the H_A row) against each distinct
  // inner-factor history q_inner[h] once into T[h] (shape (O + has_HA) x Krem,
  // Krem = K / S_inner), then finish each combination by contracting T[h_inner]
  // against the Kronecker product of the remaining outer factors' marginals
  // (Krem-sized — S_inner times smaller than the full product). Cost drops from
  // prod(H)*O*K to H_inner*O*K + prod(H)*(O*Krem + Krem-build): the S_inner
  // redundancy the per-entry path pays on the inner factor.
  //
  // For rank 2 the outer product is just q_outer[h0] (rank-1 build returns the
  // marginal pointer, no materialization), recovering the two-factor schedule
  // exactly. Scoped to the serial modality phase and the non-param-info-gain
  // case: the OMP phase (deep inductive, amortized across cores) and the pA/wA
  // novelty path keep the per-entry routine. q_outer stays f32 through stage 2
  // (the per-entry path quantizes the product to f16), so results move toward
  // the f32 JAX reference, not away from it.
  void score_modality_shared(int64_t t, const ModalityMeta& mm, const QsLevel& qs_next,
                             const FactorHistoryTables& tables, int64_t tuple_base, const int64_t* strides,
                             int64_t local_n, float* score_m, NodeScratch& s) const {
    const DependencyView& deps    = mm.state_deps;
    const int             R       = deps.rank;
    const int64_t         f_inner = deps.factors[R - 1];
    const int64_t         S_inner = L.S[f_inner];
    const int64_t         K       = mm.K;
    const int64_t         Krem    = K / S_inner;  // prod(S over outer factors)
    const int64_t         O       = mm.O;
    const int64_t         H_inner = qs_next.histories[f_inner];
    const bool            sig     = flags.use_states_info_gain;
    const int64_t         rows    = O + (sig ? 1 : 0);

    const float* qin_base = qs_next.ptrs[f_inner];

    ensure_at_least(s.mod_T_shared, std::max<int64_t>(H_inner * rows * Krem, 1));
    ensure_at_least(s.mod_qrem_f32, std::max<int64_t>(Krem, 1));
    float* T = s.mod_T_shared.data();

    // Stage 1: contract A (rows 0..O-1) and the H_A row (row O, when sig)
    // against each inner-factor history. A[r] is laid out (Krem x S_inner)
    // row-major because the inner dependency factor is A's contiguous axis.
    for (int64_t h = 0; h < H_inner; ++h) {
      const float* qin = qin_base + h * S_inner;
      float*       Th  = T + h * rows * Krem;
#if PYMDP_FFI_HAS_F16_FML
      ensure_at_least(s.mod_qin_compact, S_inner);
      pack_f32_to_f16(S_inner, qin, s.mod_qin_compact.data());
      const FfiF16* Aaug = A_aug_compact + mm.offsets.A_aug;  // (O + 1) x K, f16
      for (int64_t r = 0; r < rows; ++r) {
        sgemv_rm_compact(Krem, S_inner, Aaug + r * K, S_inner, s.mod_qin_compact.data(), Th + r * Krem);
      }
#else
      // ARMv8.0 / non-FML: contract the f32 source A directly (no A_aug), with
      // the H_A row sourced from the separate HA precompute buffer.
      const float* Am = A_data + mm.offsets.A_input;  // O x K, f32
      for (int64_t o = 0; o < O; ++o) {
        sgemv_rm_f32(Krem, S_inner, Am + o * K, S_inner, qin, Th + o * Krem);
      }
      if (sig) {
        sgemv_rm_f32(Krem, S_inner, HA->data.data() + mm.offsets.entropy, S_inner, qin, Th + O * Krem);
      }
#endif
    }

    // Stage 2: finish each combination. q_rem = Kronecker product of the outer
    // (first R-1) factors' marginals; qo = T[h_inner] . q_rem.
    const DependencyView                       prefix{R - 1, deps.factors};
    const bool                                 have_tuples = !tables.mod_history_tuples.empty();
    std::array<int32_t, kMaxFfiDependencyRank> histories{};
    for (int64_t idx = 0; idx < local_n; ++idx) {
      if (have_tuples) {
        const int32_t* tuple = tables.mod_history_tuples.data() + tuple_base + idx * kMaxFfiDependencyRank;
        for (int i = 0; i < R; ++i) histories[i] = tuple[i];
      } else {
        decode_history_tuple(idx, strides, R, histories.data());
      }
      const int32_t h_inner = histories[R - 1];
      const float*  q_rem   = build_qs_outer_history(L, prefix, qs_next, histories.data(), kMaxFfiDependencyRank,
                                                     s.interm_f32.data(), s.interm_spare.data(), s.mod_qrem_f32.data());
      const float*  Th      = T + static_cast<int64_t>(h_inner) * rows * Krem;
      float*        qo      = s.qo_plus_HA.data();
      sgemv_rm_f32(O, Krem, Th, Krem, q_rem, qo);

      float delta = 0.0f;
      if (sig) {
        const float ha_dot = sdot_f32(Krem, Th + O * Krem, q_rem);
        delta += entropy(qo, O) - ha_dot;
      }
      if (flags.use_utility) {
        delta += sdot_f32(O, qo, C_data + mm.offsets.C_input + t * O);
      }
      score_m[idx] = delta;
    }
  }
};

}  // namespace pymdp_ffi
