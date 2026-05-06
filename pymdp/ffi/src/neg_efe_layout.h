// Layout metadata and boundary validation for the neg-EFE kernel ABI.
//
// Holds the Layout view (non-owning pointers into XLA-supplied attribute
// spans), the call-shape and buffer-stride bookkeeping, and the validation
// helpers (parse, attr-span sizes, layout invariants, buffer counts, epsilon
// shape). Precomputes/metas live in neg_efe_precompute.h.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "error_helpers.h"
#include "kernel_primitives.h"

namespace pymdp_ffi {

// Maximum modality / factor dependency rank for neg-EFE and FPI FFI paths.
// MAX_FFI_DEP_RANK in pymdp/ffi/_core.py must match.
//
// neg_efe_cuda_kernels.h duplicates this numeric cap — nvcc cannot include this
// header without pulling XLA FFI through api.h.
inline constexpr int kMaxFfiDependencyRank = 8;

// Bit positions must match pymdp/ffi/_efe.py FLAG_USE_* (exported from pymdp.ffi).
constexpr int32_t     kFlagUseUtility        = 1 << 0;
constexpr int32_t     kFlagUseStatesInfoGain = 1 << 1;
constexpr int32_t     kFlagUseInductive      = 1 << 2;
constexpr int32_t     kFlagUseParamInfoGain  = 1 << 3;
constexpr const char* kEfeKernelName         = "efe_ffi";

// Host-owned payloads matching FFI int layouts (offsets, CSR-style headers).
using FfiInt64Vec = std::vector<int64_t>;
using FfiInt32Vec = std::vector<int32_t>;
// CPU neg-EFE: outer = factor index, inner = flattened float scratch for qs.
using PerFactorFloatRows = std::vector<std::vector<float>>;

struct KernelFlags {
  bool use_utility;
  bool use_states_info_gain;
  bool use_inductive;
  bool use_param_info_gain;

  static constexpr KernelFlags from_bits(int32_t bits) {
    return {
        (bits & kFlagUseUtility) != 0,
        (bits & kFlagUseStatesInfoGain) != 0,
        (bits & kFlagUseInductive) != 0,
        (bits & kFlagUseParamInfoGain) != 0,
    };
  }
};

// Non-owning views into the Span<const int64_t> attrs supplied by XLA, plus a
// few cached sums. Pointers are stable for the duration of one FFI call.
struct Layout {
  int64_t        P;
  int64_t        T;
  int64_t        F;
  int64_t        M;
  const int64_t* S;           // per-factor hidden-state dims       [F]
  const int64_t* O;           // per-modality obs dims              [M]
  const int64_t* U;           // per-factor action counts           [F]
  const int64_t* qs_off;      // prefix sums over S                 [F+1]
  const int64_t* A_off;       // prefix sums over |A_m|             [M+1]
  const int64_t* B_off;       // prefix sums over S_f * K_f * U_f   [F+1]
  const int64_t* C_off;       // prefix sums over T * O_m           [M+1]
  const int64_t* I_off;       // prefix sums over depth_f * S_f     [F+1]
  const int64_t* I_depths;    // per-factor inductive depth         [F]
  const int64_t* A_dep_flat;  // concat of A_dependencies factor ids
  const int64_t* A_dep_off;   // prefix sums over A-dep ranks       [M+1]
  const int64_t* B_dep_flat;  // concat of B_dependencies factor ids
  const int64_t* B_dep_off;   // prefix sums over B-dep ranks       [F+1]
  int64_t        qs_flat;     // sum_f S_f
};

struct CallShape {
  int64_t Bn;
  int64_t P;
  int64_t T;
  int64_t F;
  int64_t M;
};

struct BufferStrides {
  int64_t qs;
  int64_t A;
  int64_t B;
  int64_t C;
  int64_t I;
  int64_t pm;
  int64_t out;
};

struct EpsilonLayout {
  bool batched;
};

struct DependencyView {
  int            rank;
  const int64_t* factors;
};

// The 13 attribute spans XLA hands every NegEfe target. Bundled so the
// validation / make_layout / parse_neg_efe_layout signatures don't carry 13
// parallel parameters whose order is easy to mis-thread at the call site.
// Field order matches the Bind chain in xla_register.cc and the JAX
// wrapper in pymdp/ffi/_efe.py.
struct LayoutSpans {
  FfiInt64Span S;
  FfiInt64Span O;
  FfiInt64Span U;
  FfiInt64Span qs_off;
  FfiInt64Span A_off;
  FfiInt64Span B_off;
  FfiInt64Span C_off;
  FfiInt64Span I_off;
  FfiInt64Span I_depths;
  FfiInt64Span A_dep_flat;
  FfiInt64Span A_dep_off;
  FfiInt64Span B_dep_flat;
  FfiInt64Span B_dep_off;
};

inline int a_dep_rank(const Layout& L, int64_t m) {
  return static_cast<int>(L.A_dep_off[m + 1] - L.A_dep_off[m]);
}

inline int b_dep_rank(const Layout& L, int64_t f) {
  return static_cast<int>(L.B_dep_off[f + 1] - L.B_dep_off[f]);
}
// Per-slice sizes derived from the prefix-sum offset tables.
inline int64_t a_size(const Layout& L, int64_t m) {
  return L.A_off[m + 1] - L.A_off[m];
}
inline int64_t b_size(const Layout& L, int64_t f) {
  return L.B_off[f + 1] - L.B_off[f];
}
inline int64_t c_size(const Layout& L, int64_t m) {
  return L.C_off[m + 1] - L.C_off[m];
}
inline int64_t i_size(const Layout& L, int64_t f) {
  return L.I_off[f + 1] - L.I_off[f];
}

inline DependencyView modality_state_deps(const Layout& L, int64_t m) {
  return {a_dep_rank(L, m), L.A_dep_flat + L.A_dep_off[m]};
}

inline DependencyView factor_transition_deps(const Layout& L, int64_t f) {
  return {b_dep_rank(L, f), L.B_dep_flat + L.B_dep_off[f]};
}

// S[deps.factors[i]] as int. Caller must ensure 0 <= i < deps.rank.
inline int dep_state_size(const Layout& L, const DependencyView& deps, int i) {
  return static_cast<int>(L.S[deps.factors[i]]);
}

// Per-factor K_f = prod(S[B_dependencies[f]]), the size of the parent-state
// outer product the kernel contracts B[f, u] against. For factor-local B
// (B_deps[f] == [f]), K_f reduces to S_f.
inline int64_t b_K(const Layout& L, int64_t f) {
  int64_t              K    = 1;
  const DependencyView deps = factor_transition_deps(L, f);
  for (int i = 0; i < deps.rank; ++i) K *= L.S[deps.factors[i]];
  return K;
}

inline FfiError parse_call_shape(FfiS32Buf policy_matrix, FfiInt64Span O_span, CallShape* shape) {
  const FfiInt64Span pm_dims = policy_matrix.dimensions();
  if (pm_dims.size() != 3 && pm_dims.size() != 4) {
    return invalid_arg(kEfeKernelName, "policy_matrix rank must be 3 or 4, got " + std::to_string(pm_dims.size()));
  }

  const bool    batched = (pm_dims.size() == 4);
  const int     leading = batched ? 1 : 0;
  const int64_t M       = static_cast<int64_t>(O_span.size());
  if (M <= 0) return invalid_arg(kEfeKernelName, "O size must be positive");

  *shape = {
      batched ? pm_dims[0] : 1, pm_dims[leading + 0], pm_dims[leading + 1], pm_dims[leading + 2], M,
  };
  return FfiError::Success();
}

inline FfiError validate_attr_spans(const CallShape& shape, const LayoutSpans& spans) {
  const struct Check {
    const char* name;
    int64_t     actual;
    int64_t     expected;
  } checks[] = {
      {"S", static_cast<int64_t>(spans.S.size()), shape.F},
      {"O", static_cast<int64_t>(spans.O.size()), shape.M},
      {"U", static_cast<int64_t>(spans.U.size()), shape.F},
      {"qs_offsets", static_cast<int64_t>(spans.qs_off.size()), shape.F + 1},
      {"B_offsets", static_cast<int64_t>(spans.B_off.size()), shape.F + 1},
      {"B_dep_offsets", static_cast<int64_t>(spans.B_dep_off.size()), shape.F + 1},
      {"I_offsets", static_cast<int64_t>(spans.I_off.size()), shape.F + 1},
      {"I_depths", static_cast<int64_t>(spans.I_depths.size()), shape.F},
      {"A_offsets", static_cast<int64_t>(spans.A_off.size()), shape.M + 1},
      {"C_offsets", static_cast<int64_t>(spans.C_off.size()), shape.M + 1},
      {"A_dep_offsets", static_cast<int64_t>(spans.A_dep_off.size()), shape.M + 1},
  };
  for (const Check& c : checks) {
    PYMDP_TRY(check_span_size(kEfeKernelName, c.name, c.actual, c.expected));
  }
  // Flat dep payloads: validate after offsets so the trailing entry is available.
  if (spans.A_dep_off.size() > 0) {
    const int64_t expected = spans.A_dep_off[spans.A_dep_off.size() - 1];
    PYMDP_TRY(check_span_size(kEfeKernelName, "A_dep_flat", static_cast<int64_t>(spans.A_dep_flat.size()), expected));
  }
  if (spans.B_dep_off.size() > 0) {
    const int64_t expected = spans.B_dep_off[spans.B_dep_off.size() - 1];
    PYMDP_TRY(check_span_size(kEfeKernelName, "B_dep_flat", static_cast<int64_t>(spans.B_dep_flat.size()), expected));
  }
  return FfiError::Success();
}

inline Layout make_layout(const CallShape& shape, const LayoutSpans& spans) {
  Layout L{};
  L.P          = shape.P;
  L.T          = shape.T;
  L.F          = shape.F;
  L.M          = shape.M;
  L.S          = spans.S.begin();
  L.O          = spans.O.begin();
  L.U          = spans.U.begin();
  L.qs_off     = spans.qs_off.begin();
  L.A_off      = spans.A_off.begin();
  L.B_off      = spans.B_off.begin();
  L.C_off      = spans.C_off.begin();
  L.I_off      = spans.I_off.begin();
  L.I_depths   = spans.I_depths.begin();
  L.A_dep_flat = spans.A_dep_flat.begin();
  L.A_dep_off  = spans.A_dep_off.begin();
  L.B_dep_flat = spans.B_dep_flat.begin();
  L.B_dep_off  = spans.B_dep_off.begin();
  L.qs_flat    = spans.qs_off[L.F];
  return L;
}

// Defensive layout sanity-check at the FFI boundary so misconfigured Python
// inputs fail loudly instead of corrupting kernel reads.
inline FfiError validate_layout(const Layout& L) {
  if (L.F <= 0 || L.M <= 0 || L.P <= 0 || L.T <= 0) {
    return invalid_arg(kEfeKernelName, "F/M/P/T must all be positive");
  }
  if (L.qs_off[0] != 0 || L.A_off[0] != 0 || L.B_off[0] != 0 || L.C_off[0] != 0 || L.I_off[0] != 0 ||
      L.A_dep_off[0] != 0 || L.B_dep_off[0] != 0) {
    return invalid_arg(kEfeKernelName, "offset tables must start at zero");
  }
  uint64_t total_actions = 1;
  for (int64_t f = 0; f < L.F; ++f) {
    if (L.S[f] <= 0 || L.U[f] <= 0) {
      return invalid_arg(kEfeKernelName, "S[f] and U[f] must be positive");
    }
    // Tree-level dedup packs the per-timestep action encoding into 32 bits.
    const uint64_t Uf = static_cast<uint64_t>(L.U[f]);
    if (total_actions > UINT32_MAX / Uf) {
      return invalid_arg(kEfeKernelName, "action encoding overflows 32 bits (prod(U[f]) > UINT32_MAX)");
    }
    total_actions *= Uf;
    PYMDP_TRY(check_monotonic(kEfeKernelName, "qs_off", L.qs_off[f], L.qs_off[f + 1]));
    if (L.qs_off[f + 1] - L.qs_off[f] != L.S[f]) {
      return invalid_arg(kEfeKernelName, "qs_off inconsistent with S");
    }
    PYMDP_TRY(check_monotonic(kEfeKernelName, "B_off", L.B_off[f], L.B_off[f + 1]));
    PYMDP_TRY(check_monotonic(kEfeKernelName, "B_dep_off", L.B_dep_off[f], L.B_dep_off[f + 1]));
    const DependencyView deps = factor_transition_deps(L, f);
    if (deps.rank < 1 || deps.rank > kMaxFfiDependencyRank) {
      return invalid_arg(kEfeKernelName, "B_dep rank must be in [1, 8]");
    }
    int64_t K = 1;
    for (int i = 0; i < deps.rank; ++i) {
      const int64_t fi = deps.factors[i];
      if (fi < 0 || fi >= L.F) {
        return invalid_arg(kEfeKernelName, "B_dep references out-of-range factor");
      }
      K *= L.S[fi];
    }
    if (L.B_off[f + 1] - L.B_off[f] != L.S[f] * K * L.U[f]) {
      return invalid_arg(kEfeKernelName, "B_off inconsistent with S/U/deps");
    }
  }
  for (int64_t m = 0; m < L.M; ++m) {
    if (L.O[m] <= 0) {
      return invalid_arg(kEfeKernelName, "O[m] must be positive");
    }
    PYMDP_TRY(check_monotonic(kEfeKernelName, "A_off", L.A_off[m], L.A_off[m + 1]));
    PYMDP_TRY(check_monotonic(kEfeKernelName, "C_off", L.C_off[m], L.C_off[m + 1]));
    if (L.C_off[m + 1] - L.C_off[m] != L.T * L.O[m]) {
      return invalid_arg(kEfeKernelName, "C_off inconsistent with T/O");
    }
    PYMDP_TRY(check_monotonic(kEfeKernelName, "A_dep_off", L.A_dep_off[m], L.A_dep_off[m + 1]));
    const DependencyView deps = modality_state_deps(L, m);
    if (deps.rank < 1 || deps.rank > kMaxFfiDependencyRank) {
      return invalid_arg(kEfeKernelName, "A_dep rank must be in [1, 8]");
    }
    int64_t K = 1;
    for (int i = 0; i < deps.rank; ++i) {
      const int64_t fi = deps.factors[i];
      if (fi < 0 || fi >= L.F) {
        return invalid_arg(kEfeKernelName, "A_dep references out-of-range factor");
      }
      K *= L.S[fi];
    }
    if (L.A_off[m + 1] - L.A_off[m] != L.O[m] * K) {
      return invalid_arg(kEfeKernelName, "A_off inconsistent with O/deps");
    }
  }
  for (int64_t f = 0; f < L.F; ++f) {
    if (L.I_depths[f] <= 0) {
      return invalid_arg(kEfeKernelName, "I_depths[f] must be positive");
    }
    PYMDP_TRY(check_monotonic(kEfeKernelName, "I_off", L.I_off[f], L.I_off[f + 1]));
    if (L.I_off[f + 1] - L.I_off[f] != L.I_depths[f] * L.S[f]) {
      return invalid_arg(kEfeKernelName, "I_off inconsistent with depth/S");
    }
  }
  return FfiError::Success();
}

inline BufferStrides buffer_strides(const Layout& L) {
  return {
      L.qs_flat, L.A_off[L.M], L.B_off[L.F], L.C_off[L.M], L.I_off[L.F], L.P * L.T * L.F, L.P,
  };
}

inline FfiError validate_buffer_counts(const CallShape& shape, const BufferStrides& strides, FfiS32Buf policy_matrix,
                                       FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B, FfiF32Buf C, FfiF32Buf I,
                                       FfiF32Out out) {
  struct BufferCountCheck {
    const char* name;
    int64_t     actual;
    int64_t     expected;
  };
  const BufferCountCheck checks[] = {
      {"policy_matrix", static_cast<int64_t>(policy_matrix.element_count()), shape.Bn * strides.pm},
      {"qs_init", static_cast<int64_t>(qs_init.element_count()), shape.Bn * strides.qs},
      {"A", static_cast<int64_t>(A.element_count()), shape.Bn * strides.A},
      {"B", static_cast<int64_t>(B.element_count()), shape.Bn * strides.B},
      {"C", static_cast<int64_t>(C.element_count()), shape.Bn * strides.C},
      {"I", static_cast<int64_t>(I.element_count()), shape.Bn * strides.I},
      {"out", static_cast<int64_t>(out->element_count()), shape.Bn * strides.out},
  };
  for (const BufferCountCheck& c : checks) {
    PYMDP_TRY(check_count(kEfeKernelName, c.name, c.actual, c.expected));
  }
  return FfiError::Success();
}

inline FfiError validate_epsilon_layout(FfiF32Buf inductive_epsilon, const CallShape& shape, EpsilonLayout* epsilon) {
  const FfiInt64Span eps_dims = inductive_epsilon.dimensions();
  if (eps_dims.size() == 0) {
    if (inductive_epsilon.element_count() != 1) {
      return invalid_arg(kEfeKernelName, "scalar inductive_epsilon must have one element");
    }
    *epsilon = {false};
  } else if (eps_dims.size() == 1) {
    if (inductive_epsilon.element_count() != shape.Bn) {
      return invalid_arg(kEfeKernelName,
                         "batched inductive_epsilon size = " + std::to_string(inductive_epsilon.element_count()) +
                             ", expected Bn = " + std::to_string(shape.Bn));
    }
    *epsilon = {true};
  } else {
    return invalid_arg(kEfeKernelName, "inductive_epsilon rank must be 0 or 1, got " + std::to_string(eps_dims.size()));
  }
  return FfiError::Success();
}

// parse_call_shape → validate_attr_spans → make_layout → validate_layout.
// Path-specific gates run after this and before validate_neg_efe_runtime.
inline FfiError parse_neg_efe_layout(FfiS32Buf policy_matrix, const LayoutSpans& spans, CallShape* shape, Layout* L) {
  PYMDP_TRY(parse_call_shape(policy_matrix, spans.O, shape));
  PYMDP_TRY(validate_attr_spans(*shape, spans));
  *L = make_layout(*shape, spans);
  return validate_layout(*L);
}

// buffer_strides → validate_buffer_counts → validate_epsilon_layout.
// Runs after path-specific shape/feature gates.
inline FfiError validate_neg_efe_runtime(const CallShape& shape, const Layout& L, FfiS32Buf policy_matrix,
                                         FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B, FfiF32Buf C, FfiF32Buf I,
                                         FfiF32Buf inductive_epsilon, FfiF32Out out, BufferStrides* strides,
                                         EpsilonLayout* epsilon) {
  *strides = buffer_strides(L);
  PYMDP_TRY(validate_buffer_counts(shape, *strides, policy_matrix, qs_init, A, B, C, I, out));
  return validate_epsilon_layout(inductive_epsilon, shape, epsilon);
}

// When use_param_info_gain is set, Python passes flat pA/pB matching A/B strides;
// empty buffers skip that term. Mirrors checks in pymdp/ffi/_efe.py.
inline FfiError validate_neg_efe_param_info_gain_buffers(bool use_param_info_gain, int64_t Bn,
                                                         const BufferStrides& strides, int64_t pA_elements,
                                                         int64_t pB_elements) {
  if (!use_param_info_gain) return FfiError::Success();
  const bool pA_present = pA_elements > 0;
  const bool pB_present = pB_elements > 0;
  if (!pA_present && !pB_present) {
    return invalid_arg(kEfeKernelName, "use_param_info_gain=true requires at least one of pA, pB");
  }
  if (pA_present) PYMDP_TRY(check_count(kEfeKernelName, "pA", pA_elements, Bn * strides.A));
  if (pB_present) PYMDP_TRY(check_count(kEfeKernelName, "pB", pB_elements, Bn * strides.B));
  return FfiError::Success();
}

// Single-value epsilon predicate. Reused by every neg-EFE path. Mirrors the
// per-call check in pymdp.control: positive and finite (NaN/Inf rejected even
// under `-ffast-math`, which is why we use bit-level classification).
inline FfiError check_inductive_epsilon_value(float eps) {
  if (is_nonfinite_f32(eps) || eps <= 0.0f) {
    return invalid_arg(kEfeKernelName, "inductive_epsilon must be finite and positive");
  }
  return FfiError::Success();
}

// Validate every batch's inductive_epsilon. Hoisted out of the parallel
// region/forward-graph builder so workers (CPU) and the GPU graph (CUDA)
// see only valid values.
inline FfiError validate_inductive_epsilons_batched(const float* eps_base, const CallShape& shape,
                                                    EpsilonLayout epsilon) {
  const int64_t bn_check = epsilon.batched ? shape.Bn : 1;
  for (int64_t b = 0; b < bn_check; ++b) {
    PYMDP_TRY(check_inductive_epsilon_value(eps_base[b]));
  }
  return FfiError::Success();
}

}  // namespace pymdp_ffi
