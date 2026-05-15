// Host-callable launch wrappers for the CUDA neg-EFE kernels. Declarations
// only; definitions live in neg_efe_cuda_kernels.cu (compiled by nvcc).
//
// This header is the ABI boundary between:
//   * neg_efe_cuda.cc - host-side FFI glue (XLA FFI api.h consumer; can't be
//     compiled by nvcc 10.2)
//   * neg_efe_cuda_kernels.cu - device kernels and their host-side launch
//     dispatchers (compiled by nvcc; cannot include XLA FFI api.h)
//
// All wrappers return cudaError_t. The caller (neg_efe_cuda.cc) translates
// to ffi::Error via cuda_err(). Stream is passed explicitly; the caller
// owns stream lifetime. The current host path uses the default stream.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pymdp_ffi {
namespace cuda_kernels {

// Must match pymdp_ffi::kMaxFfiDependencyRank in neg_efe_layout.h (same value
// as MAX_FFI_DEP_RANK in pymdp/ffi/_core.py). Not included here: nvcc + this
// header must not pull XLA FFI api.h.
constexpr int kBRolloutMaxParents = 8;

// Per-(t, f) generalized B-rollout for factor-local or multi-parent B.
//
// Computes
//   qs_out[b, h_next, s] = sum_k B[b, s, k, action[h_next]] * qs_outer[b, h_next, k]
// where K = K_f = prod over B_deps[f] of S_parent and qs_outer is the
// Kronecker outer product of the parent factors' qs at level t-1, gathered
// per parent_histories[h_next, i]:
//   qs_outer[b, h_next, k] = prod_i qs_parents.qs[i][b, parent_histories[h_next * n_parents + i],
//                                                    s_i(k)]
// Decoding k to (s_0, ..., s_{N_PARENTS-1}) uses row-major strides over
// qs_parents.S[i] (last dim innermost).
//
// At t=0 every parent_histories entry is 0 (qs_init has implicit H=1; the
// caller supplies qs_init pointers in qs_parents.qs[i] and H[i]=1).
//
// Layout:
//   B                  [Bn, S_f, K_f, U_f]
//   action_h           [Nh]                   action u_f per (factor f's) history h_next
//   parent_histories   [Nh * n_parents]       row-major; values in [0, H_p_i)
//   qs_parents.qs[i]   [Bn, H_p_i, S_p_i]     parent factor i's qs from level t-1
//   qs_out             [Bn, Nh, S_f]
//
// One thread block per (b, h_next); shared memory holds K_f floats for
// qs_outer. The dispatcher picks the N_PARENTS template instantiation.
struct BRolloutParents {
  const float* qs[kBRolloutMaxParents];
  int          H[kBRolloutMaxParents];  // history count per parent factor at level t-1
  int          S[kBRolloutMaxParents];  // state-space size per parent factor
};

// `v_full` (when non-null and `ind_score_t_f` is also non-null) folds the
// per-(t, f) inductive score `sum_s qs_out[b, h, s] * v_full[b, qs_off_f + s]`
// into the same block as phase 2: each thread that owns an s-stripe in
// phase 2 accumulates the inductive partial alongside the factor_score
// partial; both reduce via b_rollout_block_reduce_sum_lane0. `ind_score_t_f` is
// the per-(t, f) base within inductive_concat [Bn, ind_b_stride].
cudaError_t launch_b_rollout_general(const float* B, const float* wB_tr, const float* v_full, int qs_flat, int qs_off_f,
                                     const int32_t* action_h, const int32_t* parent_histories,
                                     const BRolloutParents& parents, int n_parents, int Bn, int Nh, int S_f, int K_f,
                                     int U_f, float* qs_out, float* factor_score, float* ind_score_t_f,
                                     int64_t ind_b_stride, cudaStream_t stream);

// Per-modality factor-history scoring kernels.
//
// Rank-1: direct; rank-2/3: cuBLAS GEMM from the .cc TU + device finish here.
// Rank-3 precontracts two factors via cuBLAS, then finishes with the third.
// Outputs land in per-(t, m) slices of scores_concat; final scatter assembles them.
//
// linear[t][m][b, k] = (use_states_info_gain ? -HA[m][b, k] : 0)
//                    + (use_utility ? sum_o A[m][b, o, k] * C[t][m][b, o] : 0)
//
// A_unflat shape: [Bn, O, S_d0, ..., S_d_{rank-1}]
// linear   shape: [Bn,    S_d0, ..., S_d_{rank-1}]  (per (t, m))
// qs_d[i]  shape: [Bn, H_d_i, S_d_i]
// score_out shape: [Bn, prod_i H_d_i] slice within scores_concat
constexpr int kRankMax = 3;

// Rank-1 dedup: one thread per (b, h_d_0). score_out points to the per-
// (t, m) slice within scores_concat [Bn, total_mod]; b_stride = total_mod.
cudaError_t launch_modality_score_dedup_rank1(const float* A_unflat, const float* wA_unflat, const float* linear,
                                              const float* qs_d_0, int Bn, int O, int H_d_0, int S_d_0,
                                              int64_t b_stride, bool use_states, bool use_linear, bool use_pA,
                                              float* score_out, cudaStream_t stream);

// Rank-3 split stage 2: finish + entropy. Reads tmp_qo / tmp_lin from
// the cuBLAS pipeline and qs_split[h_split, s_split], then writes the
// (t, m) score slice. Templated on O internally.
cudaError_t launch_modality_score_dedup_rank3_stage2(const float* tmp_qo, const float* tmp_lin, const float* tmp_wa,
                                                     const float* qs_split, int Bn, int O, int H_keep_0, int H_keep_1,
                                                     int H_split, int S_split, int64_t b_stride, bool use_states,
                                                     bool use_linear, bool use_pA, float* score_out,
                                                     cudaStream_t stream);

// Per-factor inductive coefficient v_full. Equivalent to the host-side
// precompute_inductive in neg_efe_precompute.h:
//   v_f[s] = path_avail_f * log(eps) * (1 - I[f, m_f, s])
// where idx = argmax(qs_init[f]), m_f = max(argmax(I[f, :, idx]) - 1, 0),
// and path_avail_f = clip(sum_i I[f, i, idx], 0, 1). One block per (b, f);
// thread 0 of each block computes the per-factor scalars, then the warp
// writes the S_f outputs in parallel. Static factor metadata (S, depth,
// qs_off, I_off) lives in the tree cache as device-side int32 arrays.
// `eps_stride` is 0 for scalar epsilon, 1 for `[Bn]` epsilon — the kernel
// reads `eps[b * eps_stride]`.
cudaError_t launch_v_full(const float* qs_init, const float* I, const float* eps, int eps_stride, int Bn, int F,
                          int qs_flat, int64_t I_per_batch, const int32_t* S, const int32_t* depth,
                          const int32_t* qs_off, const int32_t* I_off, float* v_out, cudaStream_t stream);

// Final scatter: out[b, p] = sum_(t, m) scores_concat[b, mod_off[t,m] +
// pmi[t,m,p]] + sum_(t, f) inductive_concat[b, ind_off[t,f] + p2h[t,f,p]]
// (the inductive sum runs only when use_inductive is true). One thread per
// (b, p).
cudaError_t launch_final_scatter_dedup(const float* scores_concat, const float* inductive_concat,
                                       const float* factor_scores, const int32_t* policy_to_modality_idx,
                                       const int32_t* policy_to_factor_history, const int64_t* mod_off,
                                       const int64_t* ind_off, int Bn, int T, int M, int F, int P, int64_t total_mod,
                                       int64_t total_ind, bool use_inductive, bool use_factor_scores, float* out,
                                       cudaStream_t stream);

// ---------------- cuBLAS pipeline device helpers ----------------
//
// Rank-2 and rank-3 stage-1 share the same q01_outer build + cuBLAS GEMM
// pattern with different finish kernels. The .cc TU drives the pipeline
// (cuBLAS headers can't be included in the .cu TU).

// Build q01_outer[b, k_keep, h_kk] = qs_keep_0[h_0, s_0] * qs_keep_1[h_1, s_1].
// Used by both rank-2 (K_d = S_d_0 * S_d_1, H_kk = H_d_0 * H_d_1) and rank-3
// (K_keep = S_keep_0 * S_keep_1, H_kk = H_keep_0 * H_keep_1).
cudaError_t launch_build_qs01_outer(const float* qs_keep_0, const float* qs_keep_1, int Bn, int H_0, int H_1, int S_0,
                                    int S_1, float* q01_outer, cudaStream_t stream);

// Rank-3 stage-1 only: transpose tmp_qo_cublas[Bn, O*S_split, H_kk] to
// split_tmp_qo[Bn, H_kk, O, S_split] (the layout stage 2 expects).
cudaError_t launch_tmp_qo_cublas_to_my(const float* tmp_cublas, int Bn, int O, int S_split, int H_kk, float* tmp_my,
                                       cudaStream_t stream);

// Rank-3 stage-1 only: tmp_lin[b, h, s_split] = sum_{k_keep} q01[b, k_keep, h] *
// linear[b, k_keep * S_split + s_split].
cudaError_t launch_tmp_lin_per_h(const float* q01_outer, const float* linear, int Bn, int K_keep, int H_kk, int S_split,
                                 float* tmp_lin, cudaStream_t stream);

// Rank-2 finish (after build_qs01_outer + cublasSgemmStridedBatched
// produces tmp_qo[Bn, O, H_kk]). Computes per (b, h_kk):
//   score = -sum_o xlogx(tmp_qo[b, o, h]) + sum_k q01[b, k, h] * linear[b, k]
// One thread per (b, h_kk).
cudaError_t launch_modality_score_dedup_rank2_cublas_finish(const float* tmp_qo, const float* tmp_wa,
                                                            const float* q01_outer, const float* linear, int Bn, int O,
                                                            int H_kk, int K_d, int64_t b_stride, bool use_states,
                                                            bool use_linear, bool use_pA, float* score_out,
                                                            cudaStream_t stream);

cudaError_t launch_modality_score_dedup_rank2_fused_tiny(const float* A_unflat, const float* wA_unflat,
                                                         const float* linear, const float* qs_d_0, const float* qs_d_1,
                                                         int Bn, int O, int H_d_0, int H_d_1, int S_d_0, int S_d_1,
                                                         int64_t b_stride, bool use_states, bool use_linear,
                                                         bool use_pA, float* score_out, cudaStream_t stream);

cudaError_t launch_modality_score_dedup_rank3_fused_tiny(const float* A_unflat, const float* wA_unflat,
                                                         const float* linear, const float* qs_d_0, const float* qs_d_1,
                                                         const float* qs_d_2, int Bn, int O, int H_d_0, int H_d_1,
                                                         int H_d_2, int S_d_0, int S_d_1, int S_d_2, int64_t b_stride,
                                                         bool use_states, bool use_linear, bool use_pA,
                                                         float* score_out, cudaStream_t stream);
}  // namespace cuda_kernels
}  // namespace pymdp_ffi
