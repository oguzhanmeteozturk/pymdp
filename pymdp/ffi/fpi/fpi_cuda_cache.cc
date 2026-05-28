// Cache materialization for FpiCudaDevice: signature compute, two-tier
// (pointer-identity + FNV-1a content) miss detection, smallmeta vs
// pointer-fed dispatch staging.
//
// Holds the thread_local g_fpi_cuda_dev_scratch slot. Runtime
// (fpi_cuda_runtime.cc) calls `refresh_fpi_cuda_cache` once per call to
// either find the cache fresh or rebuild it, then reads the returned
// scratch to drive the kernel launch.

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "fpi/fpi_cuda_cache.h"
#include "fpi/fpi_cuda_context.h"
#include "fpi/fpi_cuda_kernels.h"
#include "common/cuda_memory.h"
#include "fpi/fpi_entry.h"
#include "fpi/fpi_layout.h"

// CUDA error category for this TU. `cuda_err` / `cublas_err` in
// common/cuda_memory.h take an explicit kernel name; pin it to "fpi_ffi"
// here so the FfiError surfaces as an FPI error, not a neg-EFE one.
#define CUDA_TRY(op, expr) PYMDP_TRY(::pymdp_ffi::cuda_err(::pymdp_ffi::kFpiKernelName, op, (expr)))

namespace pymdp_ffi {
namespace {

inline thread_local FpiCudaDeviceScratch g_fpi_cuda_dev_scratch;

}  // namespace

FfiError refresh_fpi_cuda_cache(FfiInt64Span S_span, FfiInt64Span ll_offsets, FfiInt64Span lp_offsets,
                                FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter, int64_t F,
                                int64_t M, cudaStream_t stream, FpiCudaDeviceScratch** scratch_out,
                                bool* use_smallmeta_out) {
  FpiCudaDeviceScratch& cs = g_fpi_cuda_dev_scratch;

  // Two-tier host-side metadata cache:
  //   1. Pointer-identity fast path: compare the five attr-span begin
  //      pointers + sizes against the last upload; on hit skip the hash.
  //   2. FNV-1a content hash (fallback) when pointers/sizes diverge.
  // A hit on either tier skips validate_fpi_attrs and the dispatch rebuild,
  // since byte-identical inputs would re-pass and re-produce the same table.
  const bool ptr_cache_hit =
      cs.sig != 0 && cs.last_S_ptr == S_span.begin() && cs.last_S_size == S_span.size() &&
      cs.last_ll_offsets_ptr == ll_offsets.begin() && cs.last_ll_offsets_size == ll_offsets.size() &&
      cs.last_lp_offsets_ptr == lp_offsets.begin() && cs.last_lp_offsets_size == lp_offsets.size() &&
      cs.last_A_dep_flat_ptr == A_dep_flat.begin() && cs.last_A_dep_flat_size == A_dep_flat.size() &&
      cs.last_A_dep_offsets_ptr == A_dep_offsets.begin() && cs.last_A_dep_offsets_size == A_dep_offsets.size();

  uint64_t sig;
  bool     host_cache_hit;
  if (ptr_cache_hit) {
    sig            = cs.sig;
    host_cache_hit = true;
  } else {
    sig = fpi_layout_signature(FpiSpans{S_span, ll_offsets, lp_offsets, A_dep_flat, A_dep_offsets}, F, M);
    // 0 is the cold-cache sentinel for cs.sig, so a hash of 0 would never
    // promote and would re-upload on every call. Bump it to a nonzero value.
    if (sig == 0) sig = 1;
    host_cache_hit = (sig == cs.sig);
  }

  if (!host_cache_hit) {
    PYMDP_TRY(validate_fpi_attrs(S_span, ll_offsets, lp_offsets, A_dep_offsets, num_iter, F, M));
  } else if (num_iter <= 0) {
    // num_iter isn't part of the cache key (it doesn't affect the dispatch
    // table), so guard it explicitly on the hit path. validate_fpi_attrs
    // covers the miss path.
    return invalid_arg(kFpiKernelName, "num_iter = " + std::to_string(num_iter) + ", must be positive");
  }

  // Pick the kernel variant up-front: small models go through the
  // cmem-parameter-bank path (LDC reads, no per-call H2D for the dispatch
  // table); larger models use the pointer-fed path (LDG reads via device
  // buffers staged on cache miss). Purely a function of F / M;
  // production rollouts (F=3, M=3) all land on smallmeta.
  const bool use_smallmeta =
      F <= static_cast<int64_t>(fpi_cuda::kMaxFSmallMeta) && M <= static_cast<int64_t>(fpi_cuda::kMaxMSmallMeta);

  if (!host_cache_hit) {
    // Build per-modality dispatch on host. Validate K range — if any
    // modality has K >= 4 the gate in _fpi.py screwed up; fail loudly
    // rather than produce silent garbage from the kernel's no-op default
    // arm.
    std::vector<int32_t>                       S_host(F);
    std::vector<int32_t>                       lp_offsets_host(F);
    std::vector<fpi_cuda::ModalityDispatchGpu> mods_host(M);
    for (int64_t f = 0; f < F; ++f) {
      S_host[f]          = static_cast<int32_t>(S_span[f]);
      lp_offsets_host[f] = static_cast<int32_t>(lp_offsets[f]);
    }
    for (int64_t m = 0; m < M; ++m) {
      const int64_t dep_start = A_dep_offsets[m];
      const int64_t K         = A_dep_offsets[m + 1] - dep_start;
      if (K < 1 || K > fpi_cuda::kRankMax) {
        return invalid_arg(kFpiKernelName, "FpiCudaDevice handles modality K in [1, " +
                                               std::to_string(fpi_cuda::kRankMax) + "]; modality " +
                                               std::to_string(m) + " has K = " + std::to_string(K));
      }
      fpi_cuda::ModalityDispatchGpu& md = mods_host[m];
      md.K                              = static_cast<int32_t>(K);
      md.ll_off                         = static_cast<int32_t>(ll_offsets[m]);
      for (int64_t i = 0; i < K; ++i) {
        const int64_t d = A_dep_flat[dep_start + i];
        if (d < 0 || d >= F) {
          return invalid_arg(kFpiKernelName, "modality " + std::to_string(m) + " references out-of-range factor");
        }
        md.Ss[i]      = S_host[d];
        md.lp_offs[i] = lp_offsets_host[d];
        // Duplicate factor within a modality aliases two log_q slices onto
        // the same offset, silently corrupting the update. can_handle_fpi
        // rejects this up front; this is the C++ safety net (mirrors
        // build_modality_dispatch).
        for (int64_t j = 0; j < i; ++j) {
          if (md.lp_offs[j] == md.lp_offs[i]) {
            return invalid_arg(kFpiKernelName,
                               "modality " + std::to_string(m) + " has duplicate factor in A_dependencies");
          }
        }
      }
      for (int64_t i = K; i < fpi_cuda::kRankMax; ++i) {
        md.Ss[i]      = 0;
        md.lp_offs[i] = 0;
      }
    }

    // Per-modality WAW sync bitmask. Bit m == 1 iff modality m's factor
    // set overlaps modality m+1's, requiring a __syncthreads() between
    // them. Bit (M-1) is always 0 — an unconditional post-loop barrier in
    // the kernel covers the final write before the convergence read /
    // next iter's softmax. M is bounded at 32 by the bitmask width;
    // production active-inference models have M well under that, and
    // restricting here keeps the kernel-side check a simple uint32_t
    // shift with no UB concerns.
    if (M > 32) {
      return invalid_arg(kFpiKernelName, "FpiCudaDevice handles at most 32 modalities; got M = " + std::to_string(M));
    }
    uint32_t mask = 0;
    for (int64_t m = 0; m + 1 < M; ++m) {
      const int64_t a_start = A_dep_offsets[m];
      const int64_t a_end   = A_dep_offsets[m + 1];
      const int64_t b_start = A_dep_offsets[m + 1];
      const int64_t b_end   = A_dep_offsets[m + 2];
      bool          overlap = false;
      for (int64_t i = a_start; i < a_end && !overlap; ++i) {
        const int64_t fa = A_dep_flat[i];
        for (int64_t j = b_start; j < b_end; ++j) {
          if (A_dep_flat[j] == fa) {
            overlap = true;
            break;
          }
        }
      }
      if (overlap) mask |= (1u << static_cast<unsigned>(m));
    }
    cs.sync_mask = mask;

    if (use_smallmeta) {
      // Pack the host-side dispatch into the by-value smallmeta struct
      // that we'll hand to launch_fpi_smallmeta as a kernel argument.
      // Pad the tail with zeros so the unused slots are deterministic.
      fpi_cuda::FpiSmallMeta& sm = cs.smallmeta;
      for (int64_t f = 0; f < F; ++f) {
        sm.S[f]          = S_host[f];
        sm.lp_offsets[f] = lp_offsets_host[f];
      }
      for (int64_t f = F; f < fpi_cuda::kMaxFSmallMeta; ++f) {
        sm.S[f]          = 0;
        sm.lp_offsets[f] = 0;
      }
      for (int64_t m = 0; m < M; ++m) {
        sm.mods[m] = mods_host[m];
      }
      for (int64_t m = M; m < fpi_cuda::kMaxMSmallMeta; ++m) {
        sm.mods[m].K      = 0;
        sm.mods[m].ll_off = 0;
        for (int i = 0; i < fpi_cuda::kRankMax; ++i) {
          sm.mods[m].Ss[i]      = 0;
          sm.mods[m].lp_offs[i] = 0;
        }
      }
    } else {
      CUDA_TRY("fpi_cuda S_dev", cs.S_dev.ensure(static_cast<std::size_t>(F) * sizeof(int32_t)));
      CUDA_TRY("fpi_cuda lp_offsets_dev", cs.lp_offsets_dev.ensure(static_cast<std::size_t>(F) * sizeof(int32_t)));
      CUDA_TRY("fpi_cuda mods_dev",
               cs.mods_dev.ensure(static_cast<std::size_t>(M) * sizeof(fpi_cuda::ModalityDispatchGpu)));

      CUDA_TRY("fpi_cuda H2D S",
               cudaMemcpyAsync(cs.S_dev.ptr, S_host.data(), static_cast<std::size_t>(F) * sizeof(int32_t),
                               cudaMemcpyHostToDevice, stream));
      CUDA_TRY("fpi_cuda H2D lp_offsets",
               cudaMemcpyAsync(cs.lp_offsets_dev.ptr, lp_offsets_host.data(),
                               static_cast<std::size_t>(F) * sizeof(int32_t), cudaMemcpyHostToDevice, stream));
      CUDA_TRY("fpi_cuda H2D mods",
               cudaMemcpyAsync(cs.mods_dev.ptr, mods_host.data(),
                               static_cast<std::size_t>(M) * sizeof(fpi_cuda::ModalityDispatchGpu),
                               cudaMemcpyHostToDevice, stream));
    }
    cs.sig = sig;
  }

  // Refresh the pointer-identity record whenever the incoming attr pointers
  // differ, so the next call can take the fast path.
  if (!ptr_cache_hit) {
    cs.last_S_ptr              = S_span.begin();
    cs.last_S_size             = S_span.size();
    cs.last_ll_offsets_ptr     = ll_offsets.begin();
    cs.last_ll_offsets_size    = ll_offsets.size();
    cs.last_lp_offsets_ptr     = lp_offsets.begin();
    cs.last_lp_offsets_size    = lp_offsets.size();
    cs.last_A_dep_flat_ptr     = A_dep_flat.begin();
    cs.last_A_dep_flat_size    = A_dep_flat.size();
    cs.last_A_dep_offsets_ptr  = A_dep_offsets.begin();
    cs.last_A_dep_offsets_size = A_dep_offsets.size();
  }

  *scratch_out       = &cs;
  *use_smallmeta_out = use_smallmeta;
  return FfiError::Success();
}

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
