// Host-callable launch wrapper for the native-CUDA FPI kernel. The .cu TU
// compiles with nvcc (no XLA FFI headers) and exposes only this thin launcher
// to fpi_cuda_runtime.cc / fpi_cuda_cache.cc.
//
// All `num_iter` fixed-point iterations run inside one kernel — no D2H/H2D,
// no per-iter host roundtrip, no stream sync; JAX ops on the same stream
// pipeline naturally.
//
// Restriction: every modality's A_dependencies rank must be in [1, 3].
// K >= 4 falls back to FpiCudaHost shim or the JAX scan reference
// (gated in pymdp/ffi/_fpi.py).

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

// Single launch covers `batch` batch elements (one block each), running
// `num_iter` iterations internally. All buffers are device pointers; caller
// manages allocation/lifetime. `S`, `lp_offsets`, `mods` are device arrays
// staged by the host glue once per call.
//
// Convergence early-exit: fires when `max |log_q[i] - log_q_prev[i]| < 1e-5f`
// after iter > 0 (matches CPU kernel tolerance). The snapshot copy is folded
// into the per-iter softmax pass — each warp writes log_q_prev while log_q
// is hot, removing a standalone copy loop and __syncthreads.
//
// `sync_mask`: bitmask over modalities; bit m == 1 means __syncthreads()
// is required after modality m because modality m+1 writes a factor slice
// that overlaps modality m's writes (without it, `log_q[f] += ...` races
// in-flight writes). Bit (M-1) is always 0; an unconditional barrier after
// the modality loop covers the final write. Computed from A_dependencies in
// fpi_cuda_cache.cc.
cudaError_t launch_fpi(const float* ll_flat,  // [batch, total_ll]
                       const float* lp_flat,  // [batch, total_S]
                       float*       q_out,    // [batch, total_S]
                       int batch, int F, int M, int total_ll, int total_S, int num_iter,
                       const int32_t*             S,           // [F]
                       const int32_t*             lp_offsets,  // [F]
                       const ModalityDispatchGpu* mods,        // [M]
                       uint32_t                   sync_mask,
                       cudaStream_t               stream);

// Diagnostic: empty kernel with the same grid/block/shmem as launch_fpi.
// Used by PYMDP_FFI_FPI_KERNEL_NOOP in FpiCudaDevice; diff vs launch_fpi
// isolates kernel-internal work from XLA-dispatch + driver overhead.
cudaError_t launch_fpi_noop(int batch, int total_S, cudaStream_t stream);

// Small-metadata variant. Carries S, lp_offsets, and the per-modality
// dispatch table by value as kernel arguments — on sm_70+ these live in
// the constant-memory parameter bank (LDC, broadcast) instead of L1-cached
// global loads (LDG). Eliminates the per-call H2D for dispatch arrays on the
// cache-miss path and the per-iteration global loads for S[f] / lp_offsets[f]
// / mods[m] on the hot path.
//
// Caps: kMaxFSmallMeta / kMaxMSmallMeta = 8. FpiSmallMeta is ~320 bytes;
// total cmem[0] ~424 bytes — well under the 32 KB per-launch limit. Models
// with F > 8 or M > 8 use the pointer-fed path via launch_fpi.
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
