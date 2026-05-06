// Precomputes and shape metadata for the neg-EFE kernel.
//
// Reusable A/B precompute builders (B transpose, H_A per modality),
// per-call inductive v_f precompute, metadata builders, Kronecker helpers,
// and the cross-call precompute cache. Layout/ABI plumbing lives in neg_efe_layout.h.
//
// Intentionally header-only — functions are small, called from exactly one
// translation unit, and marked `inline` in case that stops being true.

#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <utility>
#include <vector>

#include "neg_efe_layout.h"
#include "kernel_primitives.h"

namespace pymdp_ffi {

// Ragged flat buffer: contiguous payload + (N+1) prefix-sum offsets so that
// element i covers [offsets[i], offsets[i+1]). Shared by transposed B,
// precomputed H_A, and precomputed inductive v_f.
template <class T> struct RaggedBuffer {
  std::vector<T>       data;
  std::vector<int64_t> offsets;  // size N+1
};

// Transpose each B[f] from flat (S_f, K_f, U_f) to flat (U_f, S_f, K_f) so the
// hot loop can read every (f, u) slice as one row-major sgemv matrix. Here
// K_f = prod(S[B_dependencies[f]]); factor-local B is K_f = S_f.
//
// The transposed buffer can have a different stride than the input when
// K_f != S_f, so we can't reuse the input's L.B_off offsets here.
using TransposedB = RaggedBuffer<float>;

// Initialize a TransposedB with offsets[f+1] = offsets[f] + S_f*K_f*U_f and
// a zero-filled data buffer of that total size. Shared by transpose_B and
// precompute_wB_transposed — both target the same `(U_f, S_f, K_f)` layout.
inline void init_transposed_b(const Layout& L, TransposedB* out) {
  out->offsets.assign(L.F + 1, 0);
  for (int64_t f = 0; f < L.F; ++f) {
    const int64_t K     = b_K(L, f);
    out->offsets[f + 1] = out->offsets[f] + L.S[f] * K * L.U[f];
  }
  out->data.assign(out->offsets[L.F], 0.0f);
}

inline TransposedB transpose_B(const Layout& L, const float* B_data) {
  TransposedB out;
  init_transposed_b(L, &out);

  for (int64_t f = 0; f < L.F; ++f) {
    int64_t      S      = L.S[f];
    int64_t      K      = b_K(L, f);
    int64_t      Uf     = L.U[f];
    const float* Bf     = B_data + L.B_off[f];
    float*       Bf_out = out.data.data() + out.offsets[f];
    for (int64_t sn = 0; sn < S; ++sn) {
      for (int64_t sk = 0; sk < K; ++sk) {
        for (int64_t u = 0; u < Uf; ++u) {
          Bf_out[(u * S + sn) * K + sk] = Bf[(sn * K + sk) * Uf + u];
        }
      }
    }
  }
  return out;
}

// pymdp.maths.MINVAL — the float epsilon used to clip pA/pB before log /
// digamma so the Dirichlet weight stays finite at zeroed-out positions.
inline constexpr float kWnormMinval = std::numeric_limits<float>::epsilon();

// Per-cell hoisted inputs to the wnorm formula. Computed once per cell, then
// reused across every (o, k) / (s, k, u) entry that shares that cell.
inline void wnorm_safe_sums(const float* sums, int64_t n, float* log_out, float* inv_out, float* dig_out) {
  for (int64_t i = 0; i < n; ++i) {
    const float s = std::max(sums[i], kWnormMinval);
    log_out[i]    = std::log(s);
    inv_out[i]    = 1.0f / s;
    dig_out[i]    = digamma_f32(s);
  }
}

// Per-element Dirichlet weight from `_exact_wnorm`:
//   w(p; sum) = log(sum) - log(p) + 1/p - 1/sum + digamma(p) - digamma(sum)
// Returns 0 when `p <= 0` (matches the JAX `(p > 0)` mask that suppresses
// contributions from positions zeroed out by Bayesian model reduction). `p` is
// clipped to kWnormMinval inside; the `log_sum_v / inv_sum_v / dig_sum_v`
// triple is the hoisted cell summary from `wnorm_safe_sums`.
inline float wnorm_weight(float p, float log_sum_v, float inv_sum_v, float dig_sum_v) {
  if (p <= 0.0f) return 0.0f;
  const float ps = std::max(p, kWnormMinval);
  return log_sum_v - std::log(ps) + 1.0f / ps - inv_sum_v + digamma_f32(ps) - dig_sum_v;
}

// Param-info-gain weights derived from the Dirichlet posterior pA, matching
// pymdp.maths._exact_wnorm. Layout matches A — `(O_m, K_m)` per modality,
// packed by L.A_off — so the hot path consumes one f32 sgemv per modality
// per node and reuses `mm.offsets.A_input` for indexing.
inline std::vector<float> precompute_wA(const Layout& L, const float* pA_data) {
  std::vector<float> out(L.A_off[L.M], 0.0f);
  for (int64_t m = 0; m < L.M; ++m) {
    const int64_t O        = L.O[m];
    const int64_t A_m_size = a_size(L, m);
    const int64_t K_m      = A_m_size / O;
    const float*  pAm      = pA_data + L.A_off[m];
    float*        wAm      = out.data() + L.A_off[m];

    // Column sums sumA[k] = sum_o pAm[o, k]. K_m floats; cache-friendly to
    // accumulate first, then reuse across the per-row pass.
    std::vector<float> sumA(K_m, 0.0f);
    for (int64_t o = 0; o < O; ++o) {
      const float* row = pAm + o * K_m;
      for (int64_t k = 0; k < K_m; ++k) sumA[k] += row[k];
    }

    std::vector<float> log_sum(K_m), inv_sum(K_m), dig_sum(K_m);
    wnorm_safe_sums(sumA.data(), K_m, log_sum.data(), inv_sum.data(), dig_sum.data());

    for (int64_t o = 0; o < O; ++o) {
      const float* row_in  = pAm + o * K_m;
      float*       row_out = wAm + o * K_m;
      for (int64_t k = 0; k < K_m; ++k) {
        row_out[k] = wnorm_weight(row_in[k], log_sum[k], inv_sum[k], dig_sum[k]);
      }
    }
  }
  return out;
}

// Param-info-gain weights for pB. Mirrors precompute_wA but input layout is
// `(S_f, K_f, U_f)`, normalizing along axis 0 (S_f); column sums over (K_f,
// U_f). Output is transposed to `(U_f, S_f, K_f)` so the hot path reuses
// B-rollout indexing (`fm.offsets.B_transposed + action_vec[f] * fm.S * fm.K`).
inline TransposedB precompute_wB_transposed(const Layout& L, const float* pB_data) {
  TransposedB out;
  init_transposed_b(L, &out);

  for (int64_t f = 0; f < L.F; ++f) {
    const int64_t S   = L.S[f];
    const int64_t K   = b_K(L, f);
    const int64_t Uf  = L.U[f];
    const float*  pBf = pB_data + L.B_off[f];
    float*        wBf = out.data.data() + out.offsets[f];

    // Column sums sum[k, u] = sum_s pBf[s, k, u]. (K * Uf) floats. Layout of
    // pBf is `(S, K, U)` row-major, so summing the leading axis is one pass.
    std::vector<float> sumB(K * Uf, 0.0f);
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t k = 0; k < K; ++k) {
        const float* p_sk  = pBf + (s * K + k) * Uf;
        float*       sum_k = sumB.data() + k * Uf;
        for (int64_t u = 0; u < Uf; ++u) sum_k[u] += p_sk[u];
      }
    }

    std::vector<float> log_sum(K * Uf), inv_sum(K * Uf), dig_sum(K * Uf);
    wnorm_safe_sums(sumB.data(), K * Uf, log_sum.data(), inv_sum.data(), dig_sum.data());

    // Compute wB and write transposed to (U, S, K) layout in one pass.
    for (int64_t s = 0; s < S; ++s) {
      for (int64_t k = 0; k < K; ++k) {
        const float* p_sk = pBf + (s * K + k) * Uf;
        for (int64_t u = 0; u < Uf; ++u) {
          const int64_t i = k * Uf + u;
          // TransposedB layout: out[(u * S + s) * K + k]
          wBf[(u * S + s) * K + k] = wnorm_weight(p_sk[u], log_sum[i], inv_sum[i], dig_sum[i]);
        }
      }
    }
  }
  return out;
}

// Build H_A[m] = -sum_o xlogx(A[m, o, ...]) for an A-cache entry.
// Layout of H_A[m] matches A[m] minus the leading O axis.
using PrecomputedHA = RaggedBuffer<float>;

inline PrecomputedHA precompute_HA(const Layout& L, const float* A_data) {
  PrecomputedHA res;
  res.offsets.resize(L.M + 1);
  res.offsets[0] = 0;
  for (int64_t m = 0; m < L.M; ++m) {
    int64_t A_m_size   = a_size(L, m);
    int64_t H_m_size   = A_m_size / L.O[m];
    res.offsets[m + 1] = res.offsets[m] + H_m_size;
  }
  res.data.assign(res.offsets[L.M], 0.0f);
  for (int64_t m = 0; m < L.M; ++m) {
    int64_t      H_m_size = res.offsets[m + 1] - res.offsets[m];
    const float* Am       = A_data + L.A_off[m];
    float*       Hm       = res.data.data() + res.offsets[m];
    for (int64_t o = 0; o < L.O[m]; ++o) {
      for (int64_t i = 0; i < H_m_size; ++i) {
        Hm[i] -= xlogx(Am[o * H_m_size + i]);
      }
    }
  }
  return res;
}

// Precompute per-factor inductive coefficient vector:
//   v_f[s] = path_avail_f * log(eps) * (1 - I[f, m_f, s])
// where
//   idx      = argmax(qs_init[f])
//   m_f      = max(argmax(I[f, :, idx]) - 1, 0)
//   path_f   = clip(sum_i I[f, i, idx], 0, 1)
// Since qs_init does not depend on policy, this is a single per-call pass.
// The inductive value per policy per timestep then reduces to v_f . qs[t+1, f].
//
// `data` is contiguous over factors, indexed by L.qs_off in the consumer.
struct PrecomputedInductive {
  std::vector<float> data;
};

inline PrecomputedInductive precompute_inductive(const Layout& L, const float* qs_init, const float* I_data,
                                                 float inductive_eps) {
  PrecomputedInductive res;
  res.data.assign(L.qs_flat, 0.0f);
  const float log_eps = std::log(inductive_eps);
  for (int64_t f = 0; f < L.F; ++f) {
    int64_t S     = L.S[f];
    int64_t depth = L.I_depths[f];
    // Layout validation should keep S >= 1 and depth >= 1, but a malformed
    // layout slipping through would otherwise dereference qsf[0] / If[0]
    // against a zero-length slice (which on a packed [Bn, qs_flat] /
    // [Bn, I_flat] buffer reads into the *next* factor's data and silently
    // corrupts v_f, then propagates through inductive scoring). res.data
    // is already zero-initialized above, so an empty factor contributes 0.
    if (S <= 0 || depth <= 0) continue;
    const float* qsf = qs_init + L.qs_off[f];
    const float* If  = I_data + L.I_off[f];

    int64_t idx  = 0;
    float   best = qsf[0];
    for (int64_t s = 1; s < S; ++s) {
      if (qsf[s] > best) {
        best = qsf[s];
        idx  = s;
      }
    }

    int64_t m_f   = 0;
    float   mbest = If[0 * S + idx];
    for (int64_t i = 1; i < depth; ++i) {
      if (If[i * S + idx] > mbest) {
        mbest = If[i * S + idx];
        m_f   = i;
      }
    }
    if (m_f > 0) m_f -= 1;

    float path_avail = 0.0f;
    for (int64_t i = 0; i < depth; ++i) path_avail += If[i * S + idx];
    path_avail = std::max(0.0f, std::min(1.0f, path_avail));

    float*      vf   = res.data.data() + L.qs_off[f];
    const float coef = path_avail * log_eps;
    for (int64_t s = 0; s < S; ++s) {
      vf[s] = coef * (1.0f - If[m_f * S + s]);
    }
  }
  return res;
}

// -----------------------------------------------------------------------------
// Per-modality / per-factor metadata
// -----------------------------------------------------------------------------

struct ModalityOffsets {
  int64_t A_input;  // f32 A input buffer
  int64_t C_input;  // f32 C input buffer
  int64_t entropy;  // PrecomputedHA.data
  int64_t A_aug;    // packed [A; H_A] buffer
};

struct FactorOffsets {
  int64_t B_transposed;  // transposed B precompute buffer
};

// Per-modality metadata computed once per call. K stays int64_t — it's a
// product of state sizes that can overflow int32_t at the rank-8 ABI cap.
// O is bounded by per-modality observation count and fits int32_t comfortably.
struct ModalityMeta {
  DependencyView  state_deps;  // A_dependencies[m]
  ModalityOffsets offsets;
  int64_t         K;  // prod(S[state_deps]) -- qs_outer_compact length
  int32_t         O;  // L.O[m]
};

// Per-factor metadata computed once per call. `offsets.B_transposed` indexes the
// transposed-B buffer, whose offsets can differ from L.B_off when K_f != S_f.
// S is a single factor's state count (int32_t-safe); K stays int64_t for the
// same rank-8 overflow reason as ModalityMeta::K.
struct FactorMeta {
  DependencyView transition_deps;  // B_dependencies[f]
  FactorOffsets  offsets;
  int32_t        S;  // L.S[f]
  int64_t        K;  // prod(S[transition_deps])
};

using FactorMetaVec   = std::vector<FactorMeta>;
using ModalityMetaVec = std::vector<ModalityMeta>;
using FfiF16Vec       = std::vector<FfiF16>;

// K = prod(S[deps]); inner = K / S[deps[rank-1]] (0 for rank < 2). Used by
// build_meta_summary to size the f32 prefix scratch.
inline int64_t kron_K(const Layout& L, const DependencyView& deps) {
  int64_t K = 1;
  for (int i = 0; i < deps.rank; ++i) K *= L.S[deps.factors[i]];
  return K;
}

inline int64_t kron_inner(const Layout& L, const DependencyView& deps, int64_t K) {
  return (deps.rank >= 2) ? K / L.S[deps.factors[deps.rank - 1]] : 0;
}

inline FactorMetaVec build_factor_metas(const Layout& L, const TransposedB& Btr) {
  FactorMetaVec metas(L.F);
  for (int64_t f = 0; f < L.F; ++f) {
    FactorMeta& fm     = metas[f];
    fm.transition_deps = factor_transition_deps(L, f);
    fm.offsets         = {Btr.offsets[f]};
    fm.S               = static_cast<int32_t>(L.S[f]);
    fm.K               = kron_K(L, fm.transition_deps);
  }
  return metas;
}

inline ModalityMetaVec build_modality_metas(const Layout& L, const PrecomputedHA& HA) {
  ModalityMetaVec metas(L.M);
  int64_t         A_aug_running = 0;
  for (int64_t m = 0; m < L.M; ++m) {
    ModalityMeta& mm = metas[m];
    mm.state_deps    = modality_state_deps(L, m);
    mm.offsets       = {L.A_off[m], L.C_off[m], HA.offsets[m], A_aug_running};
    mm.O             = static_cast<int32_t>(L.O[m]);
    mm.K             = kron_K(L, mm.state_deps);
    // dep_rank == 0: no qs_outer contraction, no A_aug row. Reserve zero.
    // dep_rank >  0: O rows of A then 1 row of H_A, packed contiguously.
    if (mm.state_deps.rank > 0) A_aug_running += (mm.O + 1) * mm.K;
  }
  return metas;
}

// Per-call dispatch metadata plus the scratch-size maxes needed by run_level.
struct MetaSummary {
  ModalityMetaVec modalities;
  FactorMetaVec   factors;
  int64_t         max_K = 1;  // largest qs_outer over modalities (A path)
  int64_t         max_O = 0;  // largest O[m]
  // Shared by A and B Kronecker paths (one f32 intermediate buffer covers
  // both); take the max over modalities and factors.
  int64_t max_inner_K_any = 0;
  int64_t max_K_b         = 1;  // largest qs_outer over factors (B path)
  int64_t max_S           = 0;  // largest S[f] — wb_qs scratch in param-info-gain path
};

inline MetaSummary build_meta_summary(const Layout& L, const PrecomputedHA& HA, const TransposedB& Btr) {
  MetaSummary s;
  s.modalities = build_modality_metas(L, HA);
  for (const ModalityMeta& mm : s.modalities) {
    if (mm.K > s.max_K) s.max_K = mm.K;
    if (mm.O > s.max_O) s.max_O = mm.O;
    const int64_t inner = kron_inner(L, mm.state_deps, mm.K);
    if (inner > s.max_inner_K_any) s.max_inner_K_any = inner;
  }
  s.factors = build_factor_metas(L, Btr);
  for (const FactorMeta& fm : s.factors) {
    if (fm.K > s.max_K_b) s.max_K_b = fm.K;
    if (fm.S > s.max_S) s.max_S = fm.S;
    const int64_t inner = kron_inner(L, fm.transition_deps, fm.K);
    if (inner > s.max_inner_K_any) s.max_inner_K_any = inner;
  }
  return s;
}

// Build compact A_aug = [A; H_A] per modality. dep_rank == 0 modalities take
// zero bytes (scored directly from f32 A / HA.data in the hot path).
//
// The H_A row is packed unconditionally — independent of flags.use_states_info_gain
// — because the precompute cache key is layout/content-only. One cache entry
// has to serve both flag combinations (state_info_gain on and off); skipping
// the H_A pack for the off-case would corrupt later calls that re-enable it.
// score_modality decides per-call whether to include the H_A row in the sgemv
// (`rows = mm.O + (use_states_info_gain ? 1 : 0)`), so the unused row is
// simply not read when the flag is off — no wasted compute on the hot path,
// only `K` extra f16 stores during precompute.
//
// ARMv8.0 fallback build (PYMDP_FFI_HAS_F16_FML == 0): returns an empty
// vector. FfiF16 == float on v8.0, so A_aug would be a 2x-redundant copy of A
// that raises L2 pressure. The v8.0 score_modality path instead scores
// directly from `A` + sdot_f32(H_A, qs), same compute cost. See
// neg_efe_cpu.cc score_modality.
inline FfiF16Vec build_A_aug_compact(const Layout& L, const ModalityMetaVec& metas, const float* A_data,
                                     const PrecomputedHA& HA) {
#if !PYMDP_FFI_HAS_F16_FML
  (void)L;
  (void)metas;
  (void)A_data;
  (void)HA;
  return {};
#else
  int64_t total = 0;
  for (const ModalityMeta& mm : metas) {
    if (mm.state_deps.rank > 0) total += (mm.O + 1) * mm.K;
  }
  FfiF16Vec out(total);
  for (const ModalityMeta& mm : metas) {
    if (mm.state_deps.rank == 0) continue;
    FfiF16* dst = out.data() + mm.offsets.A_aug;
    pack_f32_to_f16(mm.O * mm.K, A_data + mm.offsets.A_input, dst);
    pack_f32_to_f16(mm.K, HA.data.data() + mm.offsets.entropy, dst + mm.O * mm.K);
  }
  return out;
#endif
}

// -----------------------------------------------------------------------------
// Cross-call precompute cache
// -----------------------------------------------------------------------------
//
// H_A, compact A_aug, and transposed B are often invariant across scan
// steps and broadcast batches. Cache them per thread by (size, layout_sig,
// content_tag). Pointer identity is intentionally not part of the key — JAX
// may move the same data or recycle an address for different data.

// Common cache key. Payload structs add HA/A_aug_compact or transposed B.
// The sampled content_tag stays in the key even when a pointer repeats: buffer
// addresses can be recycled for different same-shaped data across calls.
struct CacheKey {
  int64_t  size        = 0;
  uint64_t layout_sig  = 0;
  uint64_t content_tag = 0;
};

struct ACache : CacheKey {
  PrecomputedHA HA;
  FfiF16Vec     A_aug_compact;
};

struct BCache : CacheKey {
  TransposedB Btr;
};

// Param-info-gain wA / wB precompute caches. Mirror ACache / BCache because
// pA / pB are typically broadcast across the vmap batch in
// Agent.infer_policies — the first batch element pays the digamma+log cost,
// the rest hit the cache. Without this, batch=4 redoes the precompute 4x.
struct WACache : CacheKey {
  std::vector<float> wA;
};

struct WBCache : CacheKey {
  TransposedB wBtr;
};

// Small per-thread LRU. Slot 0 is most-recent. Eight slots cover the common
// batch_size=4 rollout path for both A and B without thrashing between batch
// elements, while keeping the cache tiny.
inline constexpr std::size_t                                   kPrecomputeCacheSlots = 8;
inline thread_local std::array<ACache, kPrecomputeCacheSlots>  g_a_cache_lru;
inline thread_local std::array<BCache, kPrecomputeCacheSlots>  g_b_cache_lru;
inline thread_local std::array<WACache, kPrecomputeCacheSlots> g_wa_cache_lru;
inline thread_local std::array<WBCache, kPrecomputeCacheSlots> g_wb_cache_lru;

// FNV-1a-style mixers — fast, no quality requirement beyond separating common
// layout/content changes with low overhead.
inline constexpr uint64_t kFNVPrime = 1099511628211ull;
inline constexpr uint64_t kFNVBasis = 14695981039346656037ull;

inline uint64_t mix64(uint64_t h, uint64_t x) {
  h ^= x;
  h *= kFNVPrime;
  return h;
}

// Mix a dependency view's rank + per-dep S values into the running hash. Shared
// by the A and B layout signatures.
inline uint64_t hash_dep_view(uint64_t h, const Layout& L, DependencyView deps) {
  h = mix64(h, static_cast<uint64_t>(deps.rank));
  for (int i = 0; i < deps.rank; ++i) {
    h = mix64(h, static_cast<uint64_t>(L.S[deps.factors[i]]));
  }
  return h;
}

inline uint64_t a_layout_sig(const Layout& L) {
  uint64_t h = kFNVBasis;
  h          = mix64(h, static_cast<uint64_t>(L.M));
  for (int64_t m = 0; m < L.M; ++m) {
    h = mix64(h, static_cast<uint64_t>(L.O[m]));
    h = hash_dep_view(h, L, modality_state_deps(L, m));
  }
  return h;
}

inline uint64_t b_layout_sig(const Layout& L) {
  uint64_t h = kFNVBasis;
  h          = mix64(h, static_cast<uint64_t>(L.F));
  for (int64_t f = 0; f < L.F; ++f) {
    h = mix64(h, static_cast<uint64_t>(L.S[f]));
    h = mix64(h, static_cast<uint64_t>(L.U[f]));
    h = hash_dep_view(h, L, factor_transition_deps(L, f));
  }
  return h;
}

// Low-cost content fingerprint for cache invalidation. It mixes a short prefix
// plus strided samples, which catches common buffer reuse / localized edits
// without scanning full A/B payloads on every rollout step.
//
// Hazard: this is a sampled fingerprint, not a hash. A caller that mutates the
// underlying buffer in-place between calls without touching any of the sampled
// positions would get a false cache hit and read stale precompute. JAX arrays
// are immutable so the production call path is safe; FFI users that pass
// in-place-mutated raw buffers must treat each mutation as a new logical input
// (e.g. by copying) to keep the cache correct.
inline constexpr int kContentTagSamples = 8;

inline uint64_t content_tag(const float* data, int64_t size) {
  if (size <= 0) return 0;
  // Belt-and-suspenders against a (nullptr, size > 0) caller: every current
  // caller gates on buffer presence upstream (in.pA != nullptr, pA_present,
  // flags.use_utility, ...) but a future caller forwarding an optional
  // buffer with its declared size would segfault inside the std::memcpy
  // below. Treat null + positive-size identically to size <= 0.
  if (data == nullptr) return 0;
  uint64_t      h      = kFNVBasis;
  const int64_t prefix = std::min<int64_t>(size, 16);
  for (int64_t idx = 0; idx < prefix; ++idx) {
    uint32_t bits;
    std::memcpy(&bits, data + idx, sizeof(bits));
    h = mix64(h, static_cast<uint64_t>(bits));
  }
  const int64_t samples[kContentTagSamples] = {
      0, (size > 1) ? 1 : 0, size / 8, size / 4, size / 2, (size * 3) / 4, (size * 7) / 8, size - 1,
  };
  for (long long idx : samples) {
    if (idx < 0) idx = 0;
    if (idx >= size) idx = size - 1;
    uint32_t bits;
    std::memcpy(&bits, data + idx, sizeof(bits));
    h = mix64(h, static_cast<uint64_t>(bits));
  }
  return h;
}

// Mix Bn into a base layout signature. Bn is part of the cache key because
// staged data has shape `[Bn, ...]` and the same underlying buffer can appear
// with different batch sizes across calls.
inline uint64_t a_sig_bn(const Layout& L, int Bn) {
  return mix64(a_layout_sig(L), static_cast<uint64_t>(Bn));
}

inline uint64_t b_sig_bn(const Layout& L, int Bn) {
  return mix64(b_layout_sig(L), static_cast<uint64_t>(Bn));
}

// Hash the policy-action shape (P, T, F, U[]). The factor-history dedup
// extends this with B/A dependency views; see factor_history_pm_sig.
inline uint64_t pm_actions_sig(const Layout& L) {
  uint64_t h = kFNVBasis;
  h          = mix64(h, static_cast<uint64_t>(L.P));
  h          = mix64(h, static_cast<uint64_t>(L.T));
  h          = mix64(h, static_cast<uint64_t>(L.F));
  for (int64_t f = 0; f < L.F; ++f) h = mix64(h, static_cast<uint64_t>(L.U[f]));
  return h;
}

// Cache-key sig for the factor-history dedup tables. Mixes pm_actions_sig
// (P/T/F/U[]), per-factor B-dep views, and per-modality A-dep views — covering
// every Layout-derived input so the cache invalidates on any agent-shape change.
inline uint64_t factor_history_pm_sig(const Layout& L) {
  uint64_t h = pm_actions_sig(L);
  for (int64_t f = 0; f < L.F; ++f) {
    h = hash_dep_view(h, L, factor_transition_deps(L, f));
  }
  for (int64_t m = 0; m < L.M; ++m) {
    h = hash_dep_view(h, L, modality_state_deps(L, m));
  }
  return h;
}

// Refresh the LRU cache and return slot 0. On miss, recompute receives the
// destination slot so callers can fill layout-specific payload fields.
template <class Cache, std::size_t N, class Recompute>
inline Cache* refresh_cache_lru(std::array<Cache, N>& arr, const float* ptr, int64_t size, uint64_t sig,
                                Recompute&& recompute) {
  const uint64_t tag = content_tag(ptr, size);
  for (std::size_t i = 0; i < N; ++i) {
    if (arr[i].size == size && arr[i].layout_sig == sig && arr[i].content_tag == tag) {
      if (i != 0) {
        Cache hit = std::move(arr[i]);
        for (std::size_t j = i; j > 0; --j) arr[j] = std::move(arr[j - 1]);
        arr[0] = std::move(hit);
      }
      arr[0].content_tag = tag;
      return &arr[0];
    }
  }
  // Miss: evict the least-recent slot and shift the remaining entries back.
  for (std::size_t j = N - 1; j > 0; --j) arr[j] = std::move(arr[j - 1]);
  recompute(arr[0]);
  arr[0].size        = size;
  arr[0].layout_sig  = sig;
  arr[0].content_tag = tag;
  return &arr[0];
}

}  // namespace pymdp_ffi
