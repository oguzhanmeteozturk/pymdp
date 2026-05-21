// Per-call runtime for the CUDA neg-EFE pass: scratch sizing, qs / inductive
// uploads (host-pack vs D2D variants), the forward driver that walks the
// (t, f, m) tree via launch.cc helpers, and an optional timing log.

#include "neg_efe/neg_efe.h"

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include "xla/ffi/api/ffi.h"

#include "common/cuda_memory.h"
#include "common/error_helpers.h"
#include "neg_efe/factor_history_tables.h"
#include "neg_efe/factor_history_tree.h"
#include "neg_efe/neg_efe_cuda_context.h"
#include "neg_efe/neg_efe_cuda_internal.h"
#include "neg_efe/neg_efe_cuda_kernels.h"
#include "neg_efe/neg_efe_layout.h"
#include "neg_efe/neg_efe_precompute.h"

#define CUDA_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cuda_err(::pymdp_ffi::kEfeKernelName, op, (expr)))
#define CUBLAS_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cublas_err(::pymdp_ffi::kEfeKernelName, op, (expr)))

namespace pymdp_ffi {
namespace {

inline FfiError ensure_cublas_handle(NegEfeContext& ctx) {
  if (ctx.cublas_handle.handle != nullptr) return FfiError::Success();
  CUBLAS_TRY("cublasCreate", cublasCreate(&ctx.cublas_handle.handle));
  return FfiError::Success();
}

QsLevel initial_qs_level(const StageCtx& stage) {
  const int F = static_cast<int>(stage.L.F);
  QsLevel   level;
  level.ptrs.resize(F);
  level.histories.assign(F, 1);
  for (int f = 0; f < F; ++f) {
    level.ptrs[f] = stage.ctx.scratch.qs_init_per_factor[f].as<const float>();
  }
  return level;
}

}  // namespace

FfiError ensure_forward_scratch(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, cudaStream_t stream,
                                bool out_is_external) {
  const ForwardDims          d                 = forward_dims(L);
  const FactorHistoryLevels& ftree             = ctx.tree_cache.factor_tree;
  const int64_t              total_mod_entries = ctx.tree_cache.total_mod_entries;
  const int64_t              total_ind_entries = ctx.tree_cache.total_ind_entries;

  // resize is a no-op when size already matches; CuArr::ensure grows lazily.
  ctx.scratch.qs_init_per_factor.resize(d.F);
  for (int slot = 0; slot < 2; ++slot) {
    ctx.scratch.qs_factor_buf[slot].resize(d.F);
  }
  for (int f = 0; f < d.F; ++f) {
    const int Sf = static_cast<int>(L.S[f]);
    CUDA_TRY("scratch qs_init", ctx.scratch.qs_init_per_factor[f].ensure(static_cast<size_t>(Bn) * Sf * sizeof(float)));
    const int    H_f_max      = static_cast<int>(max_factor_histories_for_factor(ftree, d.T, f));
    const size_t qs_buf_bytes = static_cast<size_t>(Bn) * H_f_max * Sf * sizeof(float);
    for (int slot = 0; slot < 2; ++slot) {
      CUDA_TRY("scratch qs_factor", ctx.scratch.qs_factor_buf[slot][f].ensure(qs_buf_bytes));
    }
  }
  CUDA_TRY("scratch scores_concat",
           ctx.scratch.scores_concat.ensure(static_cast<size_t>(Bn) * total_mod_entries * sizeof(float)));
  // out_is_external: the dev path scatters directly into JAX's output buffer
  // (threaded through StageCtx.out_dev_ptr), so we must not allocate or
  // ensure() the owned scratch slot.
  if (!out_is_external) {
    CUDA_TRY("scratch out_dev", ctx.scratch.out_dev.ensure(static_cast<size_t>(Bn) * d.P * sizeof(float)));
  }
  if (flags.use_inductive) {
    CUDA_TRY("scratch v_full", ctx.scratch.v_full_dev.ensure(static_cast<size_t>(Bn) * d.qsf * sizeof(float)));
    CUDA_TRY("scratch inductive_concat",
             ctx.scratch.inductive_concat.ensure(static_cast<size_t>(Bn) * total_ind_entries * sizeof(float)));
  }
  if (needs_factor_scores(ctx, flags)) {
    CUDA_TRY("scratch factor_scores",
             ctx.scratch.factor_scores.ensure(static_cast<size_t>(Bn) * total_ind_entries * sizeof(float)));
  }
  if (ctx.tree_cache.modality_tmp_qo_max_floats > 0) {
    CUDA_TRY("scratch split_tmp_qo",
             ctx.scratch.split_tmp_qo.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    if (flags.use_param_info_gain) {
      CUDA_TRY("scratch split_tmp_wa",
               ctx.scratch.split_tmp_wa.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
      CUDA_TRY("scratch tmp_wa_cublas",
               ctx.scratch.tmp_wa_cublas.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    }
    CUDA_TRY("scratch split_tmp_lin",
             ctx.scratch.split_tmp_lin.ensure(ctx.tree_cache.split_tmp_lin_max_floats * sizeof(float)));
    // Both q01_outer and tmp_qo_cublas are sized for the worst-case rank-2/3
    // modality (cache-fill pass takes the max across modalities).
    CUDA_TRY("scratch q01_outer", ctx.scratch.q01_outer.ensure(ctx.tree_cache.q01_outer_max_floats * sizeof(float)));
    CUDA_TRY("scratch tmp_qo_cublas",
             ctx.scratch.tmp_qo_cublas.ensure(ctx.tree_cache.modality_tmp_qo_max_floats * sizeof(float)));
    PYMDP_TRY(ensure_cublas_handle(ctx));
    // Skip the cublasSetStream syscall when JAX hands us the same stream as
    // last call — the JIT typically keeps the stream stable across invocations
    // of the same compiled executable, so this short-circuits to a pointer
    // compare on the steady-state path.
    if (ctx.cublas_handle.bound_stream != stream) {
      CUBLAS_TRY("cublasSetStream", cublasSetStream(ctx.cublas_handle.handle, stream));
      ctx.cublas_handle.bound_stream = stream;
    }
  }
  return FfiError::Success();
}

void upload_qs_init(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base) {
  const ForwardDims   d   = forward_dims(L);
  std::vector<float>& tmp = g_cuda_host_pack_scratch;
  for (int f = 0; f < d.F; ++f) {
    const size_t Sf = static_cast<size_t>(L.S[f]);
    pack_batched_slices(&tmp, qs_base, Bn, d.qsf, L.qs_off[f], Sf);
    std::memcpy(ctx.scratch.qs_init_per_factor[f].ptr, tmp.data(), static_cast<size_t>(Bn) * Sf * sizeof(float));
  }
}

// D2D variant of upload_qs_init for the dev path: one cudaMemcpy2DAsync per
// factor, gathering [Bn, qs_flat] strided into contiguous per-factor [Bn, S_f].
// Stays on the XLA stream — kernel reads downstream preserve ordering.
FfiError upload_qs_init_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev, cudaStream_t stream) {
  const ForwardDims d = forward_dims(L);
  for (int f = 0; f < d.F; ++f) {
    const size_t Sf = static_cast<size_t>(L.S[f]);
    CUDA_TRY("qs_init_d2d",
             cudaMemcpy2DAsync(ctx.scratch.qs_init_per_factor[f].ptr,       // dst
                               Sf * sizeof(float),                          // dst_pitch
                               qs_dev + L.qs_off[f],                        // src start
                               static_cast<size_t>(d.qsf) * sizeof(float),  // src_pitch
                               Sf * sizeof(float),                          // width_bytes
                               Bn,                                          // height (rows)
                               cudaMemcpyDeviceToDevice, stream));
  }
  return FfiError::Success();
}

// D2D variant of upload_inductive_vector: launches the v_full kernel
// directly against JAX's qs_init / I / eps device pointers (no D2H, no
// host argmax). Requires the tree cache to be filled (per-factor metadata).
FfiError upload_inductive_vector_d2d(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_dev,
                                     const float* I_dev, const float* eps_dev, EpsilonLayout epsilon,
                                     cudaStream_t stream) {
  const ForwardDims d          = forward_dims(L);
  const int         eps_stride = epsilon.batched ? 1 : 0;
  CUDA_TRY("v_full",
           cuda_kernels::launch_v_full(
               qs_dev, I_dev, eps_dev, eps_stride, Bn, d.F, d.qsf, ctx.tree_cache.I_per_batch,
               ctx.tree_cache.factor_S_dev.as<const int32_t>(), ctx.tree_cache.factor_depth_dev.as<const int32_t>(),
               ctx.tree_cache.factor_qs_off_dev.as<const int32_t>(),
               ctx.tree_cache.factor_I_off_dev.as<const int32_t>(), ctx.scratch.v_full_dev.as<float>(), stream));
  return FfiError::Success();
}

FfiError upload_inductive_vector(NegEfeContext& ctx, const Layout& L, int Bn, const float* qs_base, const float* I_base,
                                 const float* eps_base, EpsilonLayout epsilon) {
  const ForwardDims   d        = forward_dims(L);
  const size_t        n_packed = static_cast<size_t>(Bn) * d.qsf;
  std::vector<float>& v_packed = g_cuda_host_pack_scratch;
  ensure_at_least(v_packed, n_packed);
  for (int b = 0; b < Bn; ++b) {
    const float e = eps_base[epsilon.batched ? b : 0];
    PYMDP_TRY(check_inductive_epsilon_value(e));
    const PrecomputedInductive v = precompute_inductive(L, qs_base + b * d.qsf, I_base + b * L.I_off[L.F], e);
    std::memcpy(v_packed.data() + b * d.qsf, v.data.data(), d.qsf * sizeof(float));
  }
  std::memcpy(ctx.scratch.v_full_dev.ptr, v_packed.data(), n_packed * sizeof(float));
  return FfiError::Success();
}

// Drives the forward pass. Caller must have called ensure_forward_scratch,
// upload_qs_init[_d2d], and (when use_inductive) upload_inductive_vector[_d2d].
// scatter_out_dev: device-side scatter destination (host path: ctx.scratch
// .out_dev managed buffer; dev path: JAX's output device buffer).
// out_host_or_null: non-null (NegEfeCudaHost) → D2H from scatter_out_dev
// into the host out buffer + sync after scatter; nullptr (NegEfeCudaDev)
// → scatter wrote straight to JAX's output, no D2H needed.
FfiError run_forward(NegEfeContext& ctx, const Layout& L, KernelFlags flags, int Bn, float* scatter_out_dev,
                     float* out_host_or_null, cudaStream_t stream) {
  const StageCtx    stage{ctx, L, flags, Bn, stream, scatter_out_dev};
  const ForwardDims d = forward_dims(L);

  // launch_b_rollout_level folds the per-(t, f) inductive score into phase 2
  // when use_inductive is on, so there is no separate inductive launch step.
  QsLevel curr = initial_qs_level(stage);
  for (int t = 0; t < d.T; ++t) {
    QsLevel next;
    PYMDP_TRY(launch_b_rollout_level(stage, t, curr, &next));
    PYMDP_TRY(launch_modality_scores(stage, t, next));
    curr = std::move(next);
  }

  PYMDP_TRY(launch_final_scatter(stage));
  if (out_host_or_null != nullptr) {
    // D2H into host out buffer + sync (Tegra: managed memory aliases host DRAM).
    CUDA_TRY("memcpy_async_out",
             cudaMemcpyAsync(out_host_or_null, scatter_out_dev, static_cast<size_t>(Bn) * d.P * sizeof(float),
                             cudaMemcpyDefault, stream));
    CUDA_TRY("stream_sync", cudaStreamSynchronize(stream));
  }
  return FfiError::Success();
}

void log_efe_cuda_timestats_if_enabled(NegEfeContext& ctx, const Layout& L, int64_t Bn) {
  if (const char* env = std::getenv("PYMDP_FFI_TIME"); !env || env[0] != '1') return;
  int            device = -1;
  cudaError_t    rc     = cudaGetDevice(&device);
  cudaDeviceProp prop{};
  if (rc == cudaSuccess) cudaGetDeviceProperties(&prop, device);
  int64_t factor_history_total = 0;
  for (const FactorHistoryRow& tlevel : ctx.tree_cache.factor_tree) {
    for (const FactorHistoryLevel& lv : tlevel) factor_history_total += lv.n_histories;
  }
  int64_t linear_tm_total = 0;
  for (const CuArrVec& row : ctx.linear_cache.per_tm) linear_tm_total += row.size();
  std::fprintf(stderr,
               "[efe-cuda] device=%d sm_%d%d Bn=%lld P=%lld T=%lld | factor_histories=%lld "
               "(policy_steps=%lld) | A_cached=%d A_cublas_cached=%d B_cached=%d linear_tm=%lld\n",
               device, rc == cudaSuccess ? prop.major : 0, rc == cudaSuccess ? prop.minor : 0, (long long)Bn,
               (long long)L.P, (long long)L.T, (long long)factor_history_total, (long long)(L.P * L.T),
               static_cast<int>(ctx.a_cache.arrays.size()), static_cast<int>(ctx.a_cache.cublas_views.size()),
               static_cast<int>(ctx.b_cache.arrays.size()), (long long)linear_tm_total);
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
