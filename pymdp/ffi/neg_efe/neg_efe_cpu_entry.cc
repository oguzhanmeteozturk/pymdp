// NegEfeCpu ABI entry. Parses attributes, validates the call shape, then drives
// process_batch_element across the vmap batch (broadcast_all leading dim).
// Hot per-level work lives in Kernel (neg_efe_cpu_core.h); per-batch
// orchestration in process_batch_element (neg_efe_cpu_pipeline.cc).

#include <cstdint>
#include <cstdio>

#include "xla/ffi/api/c_api.h"
#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "common/kernel_primitives.h"
#include "common/omp_helpers.h"
#include "neg_efe/neg_efe.h"
#include "neg_efe/neg_efe_cpu_core.h"
#include "neg_efe/neg_efe_cpu_pipeline.h"
#include "neg_efe/neg_efe_entry.h"
#include "neg_efe/neg_efe_layout.h"

namespace ffi = ::xla::ffi;

namespace pymdp_ffi {

FfiError NegEfeCpu(FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B, FfiF32Buf C, FfiF32Buf I,
                   FfiF32Buf pA, FfiF32Buf pB, FfiF32Buf inductive_epsilon, FfiF32Out out, FfiInt64Span S_span,
                   FfiInt64Span O_span, FfiInt64Span U_span, FfiInt64Span qs_off_span, FfiInt64Span A_off_span,
                   FfiInt64Span B_off_span, FfiInt64Span C_off_span, FfiInt64Span I_off_span,
                   FfiInt64Span I_depths_span, FfiInt64Span A_dep_flat_span, FfiInt64Span A_dep_off_span,
                   FfiInt64Span B_dep_flat_span, FfiInt64Span B_dep_off_span, int32_t flags) {
  // vmap_method="broadcast_all" prepends a leading batch dim to every buffer.
  // Detect it from policy_matrix rank and iterate per instance inside one FFI
  // dispatch; A/B cache entries are shared across broadcast batch elements.
  const LayoutSpans spans =
      make_layout_spans(S_span, O_span, U_span, qs_off_span, A_off_span, B_off_span, C_off_span, I_off_span,
                        I_depths_span, A_dep_flat_span, A_dep_off_span, B_dep_flat_span, B_dep_off_span);
  ParsedCall pc;
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

  omp_fire_if(
      parallel_batches, omp_team_size_for_work_units(pc.shape.Bn), pc.shape.Bn,
      []() -> CallScratch& { return g_call_scratch; },
      [&](int64_t b, CallScratch& sc_thread) {
        FfiError err = process_batch_element(pc.L, flags, make_batch_inputs(b), sc_thread);
        if (err.failure()) {
#pragma omp critical
          {
            if (!first_error.failure()) first_error = err;
          }
        }
      });
  if (first_error.failure()) return first_error;
  return FfiError::Success();
}

}  // namespace pymdp_ffi
