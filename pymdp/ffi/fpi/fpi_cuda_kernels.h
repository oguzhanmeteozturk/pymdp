// Host-callable launch wrapper for the native-CUDA FPI kernel. Mirror of the
// neg_efe_cuda_kernels.h split: the .cu TU compiles with nvcc (no XLA FFI
// headers) and exposes only this thin launcher to the host glue in
// fpi_cuda_runtime.cc / fpi_cuda_cache.cc.
//
// Native CUDA FPI runs all `num_iter` fixed-point iterations inside one
// kernel — no D2H/H2D, no per-iter host roundtrip, no stream sync. It's the
// structural fix for the FpiCudaHost shim's pipeline break: with the kernel
// staying on the stream, JAX's surrounding ops on the same stream pipeline
// naturally instead of stalling at the per-call sync point.
//
// Current restrictions: every modality's A_dependencies rank must be in
// [1, 3]. Modalities with K >= 4 fall back to the FpiCudaHost shim or
// the JAX scan reference (gated in pymdp/ffi/_fpi.py). The K<=3 hot paths
// cover every production fixture (rollout_loop / rollout_realistic /
// rollout_learning all use A_dep rank 2 or 3).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace pymdp_ffi {
namespace fpi_cuda {

// Per-modality dispatch metadata. Mirrors fpi_precompute.h's ModalityDispatch but
// trimmed to the K<=3 hot path: kRankMax = 3, fixed-size arrays, int32
// throughout (state sizes / lp_offsets fit comfortably). One entry per
// modality, packed contiguously in a device-side array passed to the
// kernel.
constexpr int kRankMax = 3;

struct ModalityDispatchGpu {
  int32_t K;                  // 1..3
  int32_t Ss[kRankMax];       // S[A_deps[m][i]] for i in [0, K), padded with 0
  int32_t lp_offs[kRankMax];  // lp_offsets[A_deps[m][i]] for i in [0, K), padded
  int32_t ll_off;             // ll_offsets[m]
};

// Single launch covers `batch` batch elements (one block each) and runs
// `num_iter` fixed-point iterations internally. All buffers are device
// pointers; caller manages allocation/lifetime. `S`, `lp_offsets`, `mods`
// are device arrays staged by the host glue once per call.
//
// Convergence early-exit fires when `max |log_q[i] - log_q_prev[i]|` drops
// below 1e-5f after iter > 0 (matches the CPU kernel's hard-coded tolerance
// and the parity tolerance in test_fpi_ffi.py). The snapshot copy that
// feeds the check is folded into the per-iter softmax pass — each warp
// that softmaxes a factor also writes that factor's log_q slice to
// log_q_prev while log_q is hot from the softmax load. That removes the
// standalone snapshot loop + its __syncthreads, leaving only the
// block_reduce_max as per-iter convergence overhead.
// `sync_mask` is a bitmask over modalities: bit m == 1 means the kernel must
// __syncthreads() after running modality m (because modality m+1 writes a
// factor slice that overlaps with one of modality m's writes — without the
// barrier the next modality's `log_q[f] += ...` could race against the
// in-flight writes of modality m). Bit (M-1) is always 0; an unconditional
// barrier after the modality loop covers the final write before the
// convergence read / next iter's softmax. Computed on the host in
// fpi_cuda_cache.cc from A_dependencies. For fixtures where every
// modality overlaps the next (the typical production rollout), sync_mask =
// (1u << (M-1)) - 1 and behavior matches the pre-elision design.
cudaError_t launch_fpi(const float* ll_flat,  // [batch, total_ll]
                       const float* lp_flat,  // [batch, total_S]
                       float*       q_out,    // [batch, total_S]
                       int batch, int F, int M, int total_ll, int total_S, int num_iter,
                       const int32_t*             S,           // [F]
                       const int32_t*             lp_offsets,  // [F]
                       const ModalityDispatchGpu* mods,        // [M]
                       uint32_t                   sync_mask,
                       cudaStream_t               stream);

// Diagnostic: launches an empty kernel with the same grid/block/shmem
// footprint as launch_fpi, no buffer touches and no compute. Used by the
// PYMDP_FFI_FPI_KERNEL_NOOP env-gated bypass in FpiCudaDevice to bound the
// XLA-dispatch + cuLaunchKernel + driver-roundtrip overhead share of fpi
// fixture wall time. Diff vs launch_fpi at the same fpi_large shape =
// kernel-internal work + nothing else.
cudaError_t launch_fpi_noop(int batch, int total_S, cudaStream_t stream);

// Small-metadata variant. Carries S, lp_offsets, and the per-modality
// dispatch table by value as kernel arguments — on sm_70+ these live
// in the dedicated constant-memory parameter bank, served via
// LDC (broadcast to all 32 lanes in one cycle) instead of L1-cached LDG
// against device pointers (10-20 cycles even on hit). Removes the per-
// call H2D for the dispatch arrays on the cache-miss path AND the
// per-iteration global loads for `S[f]` / `lp_offsets[f]` / `mods[m]`
// on the hot path.
//
// Capacity caps (kMaxFSmallMeta / kMaxMSmallMeta = 8 each) match the
// kernel-arg constant-memory budget on sm_70+ comfortably (FpiSmallMeta
// is ~320 bytes; current kernel cmem[0] is ~424 bytes, so we land
// well under the 32 KB per-launch limit). Models with F > 8 or M > 8
// take the existing pointer-fed path via launch_fpi.
constexpr int kMaxFSmallMeta = 8;
constexpr int kMaxMSmallMeta = 8;

struct FpiSmallMeta {
  int32_t             S[kMaxFSmallMeta];           // padded with 0 above F
  int32_t             lp_offsets[kMaxFSmallMeta];  // padded with 0 above F
  ModalityDispatchGpu mods[kMaxMSmallMeta];        // padded with K=0 above M
};

cudaError_t launch_fpi_smallmeta(const float* ll_flat, const float* lp_flat, float* q_out, int batch, int F, int M,
                                 int total_ll, int total_S, int num_iter, const FpiSmallMeta& meta,
                                 uint32_t sync_mask, cudaStream_t stream);

}  // namespace fpi_cuda
}  // namespace pymdp_ffi
