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

// Device source pointers for the current call's model inputs. The CUDA-Dev warm
// check (caches_are_warm) compares these against the recorded cold-fill sources
// (warm_devs) for the kFlagModelParamsStatic devptr fast path. nullptr marks an
// absent optional buffer (C / pA / pB).
struct DevSrcs {
  const void* pm = nullptr;
  const void* A  = nullptr;
  const void* B  = nullptr;
  const void* C  = nullptr;
  const void* pA = nullptr;
  const void* pB = nullptr;
};

// Shared device-cache state: a content/layout key, the backing pool, and the
// per-modality/factor device slices. `match()` and `clear()` are common to
// every model-param cache; the two concrete caches below add their own
// `store()` (differing only in arity) plus any extra parallel arrays.
struct CudaCacheBase {
  CudaCacheKey key;
  CuPool       pool;
  CuArrVec     arrays;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const { return key.match(tag, sz, sig) && !arrays.empty(); }
  void clear() {
    pool.reset();
    arrays.clear();
    key = CudaCacheKey{};
  }
};

// Plain array cache (B / wB): pool + per-factor slices, nothing else.
struct CudaArrayCache : CudaCacheBase {
  void store(uint64_t tag, int64_t sz, uint64_t sig, CuPool built_pool, CuArrVec built) {
    pool   = std::move(built_pool);
    arrays = std::move(built);
    key.set(tag, sz, sig);
  }
};

// A-shaped cache (A / wA): adds the rank-3 cuBLAS-permuted views that run
// parallel to `arrays` (a default-constructed slot for non-rank-3 modalities).
struct CudaACache : CudaCacheBase {
  CuArrVec cublas_views;

  void store(uint64_t tag, int64_t sz, uint64_t sig, CuPool built_pool, CuArrVec built, CuArrVec built_cublas) {
    pool         = std::move(built_pool);
    arrays       = std::move(built);
    cublas_views = std::move(built_cublas);
    key.set(tag, sz, sig);
  }
  void clear() {
    CudaCacheBase::clear();
    cublas_views.clear();
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
  int64_t     total_mod_entries = 0;
  int64_t     total_ind_entries = 0;

  FfiInt32Vec mod_h_dims;
  size_t      modality_tmp_qo_max_floats = 0;
  size_t      split_tmp_lin_max_floats   = 0;
  size_t      q01_outer_max_floats       = 0;
  size_t      stacked_lin_in_max_floats  = 0;
  size_t      stacked_lin_out_max_floats = 0;

  CuArr   factor_S_dev;
  CuArr   factor_depth_dev;
  CuArr   factor_qs_off_dev;
  CuArr   factor_I_off_dev;
  int64_t I_per_batch = 0;

  bool match(uint64_t tag, int64_t sz, uint64_t sig) const { return key.match(tag, sz, sig) && !factor_tree.empty(); }
};

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
  // Dependency-group batched rank-2 linear term: stacked_lin_in[Bn, G, K_d]
  // gathers the group's linear vectors, one cuBLAS GEMM writes
  // stacked_lin_out[Bn, G, H_kk]; each modality's finish reads its G-row slice.
  CuArr stacked_lin_in;
  CuArr stacked_lin_out;
  CuArr factor_scores;
  CuArr out_dev;
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

  // Device source pointers captured at the last successful cold fill (the param
  // instance currently resident in the caches). Used only by the
  // kFlagModelParamsStatic warm-check fast path: when the caller guarantees the
  // params are not mutated in place, devptr identity against these is a correct,
  // sync-free warm test (the sole failure mode of devptr identity is in-place
  // mutation, which the flag rules out). `warm_devs_valid` is false until a cold
  // fill populates it and is reset whenever the caches are torn down.
  DevSrcs warm_devs;
  bool    warm_devs_valid = false;

  // Monotonic sentinel used as the content_tag for caches rebuilt by the device
  // repack path (prepare_caches_device). Each learning rebuild gets a distinct
  // key so any later warm check misses and rebuilds rather than risking a stale
  // hit; the device path itself never consults these tags (it force-rebuilds).
  uint64_t repack_counter = 0;
};

}  // namespace pymdp_ffi

#endif  // PYMDP_FFI_HAS_CUDA
