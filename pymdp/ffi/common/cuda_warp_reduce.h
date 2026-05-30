// Shared CUDA warp and block reduction primitives for neg-EFE and FPI.
// Header-only; mark everything inline to avoid ODR clashes across .cu files.

#pragma once

#include <cuda_runtime.h>

namespace pymdp_ffi {
namespace cuda_kernels {

// Warp sum reduction via XOR-butterfly shuffle; used by neg-EFE and FPI.
// __shfl_xor_sync covers all supported arches (sm_53+); no pre-Volta branch needed.
__device__ __forceinline__ float warp0_shfl_sum(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 16);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 2);
  v += __shfl_xor_sync(0xffffffffu, v, 1);
  return v;
}

// Generic warp-level reduction via XOR-butterfly shuffle over 32 lanes.
// Op must be a callable with operator()(float, float) → float.
template <typename Op> __device__ __forceinline__ float warp_reduce(float v, Op op) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) {
    v = op(v, __shfl_xor_sync(0xffffffffu, v, off));
  }
  return v;
}

struct MaxOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return fmaxf(a, b); }
};

struct SumOp {
  __device__ __forceinline__ float operator()(float a, float b) const { return a + b; }
};

__device__ __forceinline__ float warp_reduce_max(float v) {
  return warp_reduce(v, MaxOp());
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
  return warp_reduce(v, SumOp());
}

// Block-wide reduction via warp-leader pattern with single __syncthreads.
// Reduces N values in parallel; result valid on lane 0 only.
template <int BLOCK_SIZE, int N_VALS> __device__ __forceinline__ void block_reduce_sum_lane0(float (&vals)[N_VALS]) {
  static_assert(BLOCK_SIZE % 32 == 0, "BLOCK_SIZE must be a multiple of warp size");
  constexpr int    NUM_WARPS = BLOCK_SIZE / 32;
  __shared__ float buf[N_VALS][NUM_WARPS];

  const int tid  = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;

#pragma unroll
  for (int i = 0; i < N_VALS; ++i) {
    vals[i] = warp0_shfl_sum(vals[i]);
  }

  if (lane == 0) {
#pragma unroll
    for (int i = 0; i < N_VALS; ++i) {
      buf[i][warp] = vals[i];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int i = 0; i < N_VALS; ++i) {
      float v = (lane < NUM_WARPS) ? buf[i][lane] : 0.0f;
      vals[i] = warp0_shfl_sum(v);
    }
  }
}

template <int BLOCK_SIZE> __device__ __forceinline__ float block_reduce_sum_lane0(float val) {
  float vals[1] = {val};
  block_reduce_sum_lane0<BLOCK_SIZE, 1>(vals);
  return vals[0];
}

template <int BLOCK_SIZE> __device__ __forceinline__ void block_reduce_sum_pair_lane0(float& a, float& b) {
  float vals[2] = {a, b};
  block_reduce_sum_lane0<BLOCK_SIZE, 2>(vals);
  a = vals[0];
  b = vals[1];
}

}  // namespace cuda_kernels
}  // namespace pymdp_ffi
