// Single-source FPI early-stop tolerance, shared by the CPU and CUDA paths.
//
// Kept in its own header (no kernel_primitives.h / XLA FFI dependency) so the
// nvcc-compiled fpi_cuda_kernels.cu can include it alongside the C++17
// convergence_helpers.h that the CPU runner uses — the two paths must agree on
// this threshold for cross-platform parity, and a literal in each TU drifts
// silently.

#pragma once

namespace pymdp_ffi {

// FPI early-stop tolerance: after each body iter the kernel computes
// `max|log_q - log_q_prev|` and breaks once below this. Empirically chosen at
// 1e-5 — well below test_fpi_ffi.py's parity atol of 1e-6 (max observed
// |q - q_ref| was 2.7e-7 across all fixtures), still loose enough to fire on
// small/easy shapes (fpi_inference, fpi_high_rank). Larger problems (fpi_large)
// don't converge to this threshold within typical num_iter=16 and run the full
// loop. Cost when not firing: one max-abs-diff pass per iter, ~5% of per-iter
// cost — recouped many times over when it does.
constexpr float kFpiConvergenceTol = 1e-5f;

}  // namespace pymdp_ffi
