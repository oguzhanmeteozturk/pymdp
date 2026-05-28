// Per-JIT-instance CUDA state for the neg-EFE kernel: model-parameter caches
// (tree, A, B, wA, wB, linear), per-call scratch, and cuBLAS handle.
// NegEfeContext is the pImpl backing NegEfeState in neg_efe.h; it lives for
// the lifetime of the compiled XLA executable.

#pragma once

#ifdef PYMDP_FFI_HAS_CUDA

#include <cstdint>
#include <utility>
#include <vector>

#include <cublas_v2.h>

#include "neg_efe/factor_history_tree.h"
#include "common/cuda_memory.h"

namespace pymdp_ffi {

// `CuArrVec`: ordered device slices (modalities, factors, batch views, ...).
// `CuArrGrid2D`: rectangular storage with struct-specific [outer][inner] meaning
// (e.g. per-(t,f) history buffers or per-(t,m) linear tensors).
using CuArrVec    = std::vector<CuArr>;
using CuArrGrid2D = std::vector<CuArrVec>;

struct CudaCacheKey {
  uint64_t content_tag = 0;
  uint64_t layout_sig  = 0;
  int64_t  size        = 0;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const {
    return content_tag == tag && size == sz && layout_sig == sig;
  }
  void set(uint64_t tag, int64_t sz, uint64_t sig) {
    content_tag = tag;
    size        = sz;
    layout_sig  = sig;
  }
};

// Device source pointers for the current call's model inputs. Consumed by the
// CUDA-Dev warm check (caches_are_warm) as the gather sources for the content
// fingerprint. nullptr marks an absent optional buffer (C / pA / pB).
struct DevSrcs {
  const void* pm = nullptr;
  const void* A  = nullptr;
  const void* B  = nullptr;
  const void* C  = nullptr;
  const void* pA = nullptr;
  const void* pB = nullptr;
};

struct CudaArrayCache {
  CudaCacheKey key;
  CuPool       pool;
  CuArrVec     arrays;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const { return key.match(tag, sz, sig) && !arrays.empty(); }
  void store(uint64_t tag, int64_t sz, uint64_t sig, CuPool built_pool, CuArrVec built) {
    pool   = std::move(built_pool);
    arrays = std::move(built);
    key.set(tag, sz, sig);
  }
  void clear() {
    pool.reset();
    arrays.clear();
    key = CudaCacheKey{};
  }
};

struct CudaACache {
  CudaCacheKey key;
  CuPool       pool;
  CuArrVec     arrays;
  CuArrVec     cublas_views;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const { return key.match(tag, sz, sig) && !arrays.empty(); }
  void store(uint64_t tag, int64_t sz, uint64_t sig, CuPool built_pool, CuArrVec built, CuArrVec built_cublas) {
    pool         = std::move(built_pool);
    arrays       = std::move(built);
    cublas_views = std::move(built_cublas);
    key.set(tag, sz, sig);
  }
  void clear() {
    pool.reset();
    arrays.clear();
    cublas_views.clear();
    key = CudaCacheKey{};
  }
};

struct CudaLinearCache {
  uint64_t    a_tag      = 0;
  uint64_t    c_tag      = 0;
  uint64_t    layout_sig = 0;
  int32_t     flags      = 0;
  bool        valid      = false;
  CuPool      pool;
  CuArrGrid2D per_tm;

  bool match(uint64_t a, uint64_t c, uint64_t sig, int32_t fl) const {
    return valid && a_tag == a && c_tag == c && layout_sig == sig && flags == fl;
  }
  void store_empty(uint64_t a, uint64_t c, uint64_t sig, int32_t fl) {
    a_tag      = a;
    c_tag      = c;
    layout_sig = sig;
    flags      = fl;
    valid      = true;
    pool.reset();
    per_tm.clear();
  }
};

struct FactorHistoryCache {
  CudaCacheKey key;
  CuPool       pool;

  FactorHistoryLevels factor_tree;
  CuArrGrid2D         factor_parent_history;
  CuArrGrid2D         factor_action_per_history;

  CuArr       policy_to_modality_idx_dev;
  CuArr       factor_policy_to_history_dev;
  FfiInt64Vec mod_score_offsets;
  FfiInt64Vec ind_score_offsets;
  CuArr       mod_score_offsets_dev;
  CuArr       ind_score_offsets_dev;
  int64_t     total_mod_entries    = 0;
  int64_t     total_ind_entries    = 0;
  int         H_max_per_factor_max = 0;

  FfiInt32Vec mod_h_dims;
  size_t      modality_tmp_qo_max_floats = 0;
  size_t      split_tmp_lin_max_floats   = 0;
  size_t      q01_outer_max_floats       = 0;

  CuArr   factor_S_dev;
  CuArr   factor_depth_dev;
  CuArr   factor_qs_off_dev;
  CuArr   factor_I_off_dev;
  int64_t I_per_batch = 0;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const { return key.match(tag, sz, sig) && !factor_tree.empty(); }
};

using CudaTreeCache = FactorHistoryCache;

struct CudaScratch {
  CuArrVec qs_init_per_factor;
  CuArr    v_full_dev;
  CuArrVec qs_factor_buf[2];
  CuArr    scores_concat;
  CuArr    inductive_concat;
  CuArr    split_tmp_qo;
  CuArr    split_tmp_wa;
  CuArr    split_tmp_lin;
  CuArr    q01_outer;
  CuArr    tmp_qo_cublas;
  CuArr    tmp_wa_cublas;
  CuArr    factor_scores;
  CuArr    out_dev;
  // Content-tag gather scratch for the CUDA-Dev warm check. Managed memory
  // so the host can read the gathered 32-bit words after one stream sync.
  // Sized for the 6 fingerprinted buffers (pm, A, B, C, pA, pB) at
  // kContentTagTotalSamples 4-byte words each = 576 bytes.
  CuArr    content_tag_dev;
};

inline thread_local std::vector<float> g_cuda_host_pack_scratch;
inline thread_local std::vector<float> g_cuda_host_pack_scratch_alt;

struct CublasHandleHolder {
  cublasHandle_t handle       = nullptr;
  cudaStream_t   bound_stream = nullptr;  // last stream cublasSetStream-set
  ~CublasHandleHolder() {
    // Mirror CuArr::reset's process-exit guard — see cuda_runtime_unloading
    // in common/cuda_memory.h for the Tegra teardown-race rationale.
    if (handle && !cuda_runtime_unloading()) cublasDestroy(handle);
  }
};

struct NegEfeContext {
  FactorHistoryCache tree_cache;
  CudaACache         a_cache;
  CudaArrayCache     b_cache;
  CudaACache         wa_cache;
  CudaArrayCache     wb_cache;
  CudaLinearCache    linear_cache;
  CudaScratch        scratch;
  CublasHandleHolder cublas_handle;
};

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
