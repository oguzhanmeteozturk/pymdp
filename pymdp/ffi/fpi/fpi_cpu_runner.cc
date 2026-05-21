// CPU FPI driver: owns the thread_local FpiScratch, builds the per-call
// dispatch table, runs the OMP-gated batch loop over fpi_one_batch, and
// hosts the FpiCpu ABI entry. Also exposes `run_fpi_kernel_host` so the
// FpiCudaHost shim (fpi_cuda_host_shim.cc) can reuse the validated CPU
// pipeline on aliased / D2H'd host buffers.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "common/kernel_primitives.h"
#include "common/omp_helpers.h"
#include "common/scratch_arena.h"
#include "fpi/fpi.h"
#include "fpi/fpi_cpu_core.h"
#include "fpi/fpi_cpu_runner.h"
#include "fpi/fpi_entry.h"
#include "fpi/fpi_precompute.h"

namespace pymdp_ffi {
namespace {

// Per-call scratch with two distinct usage patterns:
//
//   1. mods (dispatch table): built once on the master thread BEFORE the
//      parallel region. Workers access the master's mods via the const
//      pointer passed into fpi_one_batch — TLS is regular memory and the
//      OMP barrier ensures the master outlives all readers.
//   2. log_q / q / t01 / prefix / suffix (compute scratch): each OMP worker
//      calls ensure_buffers on its OWN g_fpi_scratch inside the parallel
//      region and uses its own buffers. No false sharing or cross-worker
//      writes.
//
// Resize-up-only via ScratchBuffer<float>; pymdp's call patterns repeat the
// same shapes for many calls, so the steady-state allocation count is zero
// after warm-up — that holds per-thread too, since libgomp pools workers
// across calls.
struct FpiScratch {
  ScratchBuffer<float>          log_q;
  ScratchBuffer<float>          log_q_prev;  // convergence-check snapshot of prior iter's log_q
  ScratchBuffer<float>          q;
  ScratchBuffer<float>          t01;     // K=3 shared prefix (still used for marg0 sgemv)
  ScratchBuffer<float>          prefix;  // K>=4 F-chain storage (extended in-place)
  ScratchBuffer<float>          suffix;  // K>=4 suffix_q tensors back-to-back
  std::vector<ModalityDispatch> mods;

  void ensure_dispatch(int64_t M) { ensure_at_least(mods, M); }
  void ensure_buffers(int64_t total_S, int64_t max_t01, int64_t max_prefix, int64_t max_suffix) {
    log_q.ensure(total_S);
    log_q_prev.ensure(total_S);
    q.ensure(total_S);
    t01.ensure(max_t01);
    prefix.ensure(max_prefix);
    suffix.ensure(max_suffix);
  }

  FpiScratchPtrs as_ptrs() {
    return {log_q.data(), log_q_prev.data(), q.data(), t01.data(), prefix.data(), suffix.data()};
  }
};
inline thread_local FpiScratch g_fpi_scratch{};

}  // namespace

// Shared kernel body. Both FpiCpu (platform="cpu", host buffers) and
// FpiCudaHost (platform="CUDA", device buffers aliased or D2H'd into host
// scratch) funnel through here so the validated K=1/2/3 hot paths + K>=4
// forward-chain generic path + batch OMP path are written exactly once.
// Inputs/outputs are raw host pointers + counts.
FfiError run_fpi_kernel_host(const float* ll_flat, int64_t ll_count, const float* lp_flat, int64_t lp_count,
                             float* q_out, int64_t q_count, FfiInt64Span S, FfiInt64Span ll_offsets,
                             FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets,
                             int32_t num_iter) {
  const int64_t F = static_cast<int64_t>(S.size());
  const int64_t M = static_cast<int64_t>(A_dep_offsets.size()) - 1;
  PYMDP_TRY(validate_fpi_attrs(S, ll_offsets, lp_offsets, A_dep_offsets, num_iter, F, M));

  const int64_t total_S  = lp_offsets[F];
  const int64_t total_ll = ll_offsets[M];
  if (total_S <= 0 || lp_count <= 0 || lp_count % total_S != 0) {
    return invalid_arg(kFpiKernelName, "lp_flat size = " + std::to_string(lp_count) +
                                           " not divisible by total_S = " + std::to_string(total_S));
  }
  const int64_t batch = lp_count / total_S;
  PYMDP_TRY(check_count(kFpiKernelName, "ll_flat", ll_count, batch * total_ll));
  PYMDP_TRY(check_count(kFpiKernelName, "q_out", q_count, batch * total_S));

  // Caller-validated invariants — restate so the optimizer can drop zero-trip
  // guards on the per-batch / per-factor / per-modality loops below.
  if (F <= 0 || M <= 0 || batch <= 0 || total_S <= 0 || num_iter <= 0) __builtin_unreachable();

  FpiScratch& sc = g_fpi_scratch;
  sc.ensure_dispatch(M);
  int64_t max_t01 = 0, max_prefix = 0, max_suffix = 0;
  PYMDP_TRY(build_modality_dispatch(S, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets, F, M, &sc.mods, &max_t01,
                                    &max_prefix, &max_suffix));

  KernelTimer timer([&](double us) {
    std::fprintf(stderr, "[fpi] batch=%lld F=%lld M=%lld iter=%d total_S=%lld | total=%6.2f us\n",
                 static_cast<long long>(batch), static_cast<long long>(F), static_cast<long long>(M),
                 static_cast<int>(num_iter), static_cast<long long>(total_S), us);
  });

  // Dispatch one batch element. Body is identical between the serial and
  // parallel paths; omp_fire_if handles the structural serial/parallel split
  // that dodges libomp's 1-thread-team overhead on the small-batch path.
  const ModalityDispatch* mods_ptr = sc.mods.data();
  const int64_t           ni       = num_iter;
  omp_fire_if(
      should_parallelize_fpi_batch(batch, ni, total_ll), omp_team_size_for_work_units(batch), batch,
      [&]() {
        g_fpi_scratch.ensure_buffers(total_S, max_t01, max_prefix, max_suffix);
        return g_fpi_scratch.as_ptrs();
      },
      [&](int64_t b, const FpiScratchPtrs& sc_ptrs) {
        fpi_one_batch(ni, F, M, total_S, S.begin(), lp_offsets.begin(), lp_flat + b * total_S, ll_flat + b * total_ll,
                      mods_ptr, q_out + b * total_S, sc_ptrs);
      });

  return FfiError::Success();
}

// =============================================================================
// ABI entry point
// =============================================================================

FfiError FpiCpu(FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S, FfiInt64Span ll_offsets,
                FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter) {
  return run_fpi_kernel_host(ll_flat.typed_data(), ll_flat.element_count(), lp_flat.typed_data(),
                             lp_flat.element_count(), q_out->typed_data(), q_out->element_count(), S, ll_offsets,
                             lp_offsets, A_dep_flat, A_dep_offsets, num_iter);
}

}  // namespace pymdp_ffi
