"""Python wrapper for the fused neg-EFE FFI kernel."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._core import (
    _register,
    _register_multistage,
    cuda_platform_allowed_by_env,
    dep_indices_in_range,
    dep_list_rank_ok,
    ffi_capabilities,
    flatten_dep_lists,
    flatten_ragged_f32,
    has_jax_cuda_backend,
    is_available,
)

FLAG_USE_UTILITY = 1 << 0
FLAG_USE_STATES_INFO_GAIN = 1 << 1
FLAG_USE_INDUCTIVE = 1 << 2
FLAG_USE_PARAM_INFO_GAIN = 1 << 3
# Caller guarantees the model params (A, B, C, pA, pB, policy_matrix) are not
# mutated in place across calls (no learning step rewrites the donated A/B
# buffers). Lets the CUDA-Dev warm check trust devptr identity and reuse the
# resident caches. Clear (the default) means "no such guarantee": the CUDA-Dev
# cold path rebuilds the A/B/linear caches from the device buffers every call
# (always correct — never stale), which is also the learning regime. No effect
# on the CPU/host path (it keys caches on a host content tag).
FLAG_MODEL_PARAMS_STATIC = 1 << 4

_NEG_EFE_STATE_TYPE_NAME = "pymdp_neg_efe_state"
_NEG_EFE_STATE_TYPE_ID = "pymdp_neg_efe_state_type_id"
_NEG_EFE_STATE_TYPE_INFO = "pymdp_neg_efe_state_type_info"

# The CUDA modality-scoring path only has rank-1/2/3 kernels (see
# neg_efe_cuda_launch.cc::launch_modality_scores); rank>=4 hard-errors. The CPU
# path handles arbitrary rank, so this cap only gates the CUDA target — rank>=4
# A-deps fall back to JAX (CUDA host) / CPU FFI (CPU host).
CUDA_MODALITY_MAX_RANK = 3


def _register_neg_efe_state_multistage(
    *,
    target_name: str,
    instantiate_symbol: str,
    execute_symbol: str,
    platform: str,
) -> None:
    """Register a CUDA NegEfe multi-stage target sharing ``NegEfeState`` FFI type."""
    _register_multistage(
        target_name,
        instantiate_symbol=instantiate_symbol,
        execute_symbol=execute_symbol,
        type_name=_NEG_EFE_STATE_TYPE_NAME,
        type_id_symbol=_NEG_EFE_STATE_TYPE_ID,
        type_info_symbol=_NEG_EFE_STATE_TYPE_INFO,
        platform=platform,
    )


def _register_all_efe_targets_eager() -> None:
    """Register every neg-EFE FFI target the build can offer, at import time.

    Lazy registration (inside ``neg_efe_all_policies_ffi``) fires
    ``jax.ffi.register_ffi_type`` mid-JIT-trace on the first call. On Orin
    that perturbs the FFI TypeRegistry's internal storage *after* Mosaic-GPU
    has already cached a reference into it for its softmax / device-handle
    CustomCalls — the cached reference goes stale, and the next destructor
    of a Mosaic-emitted ExecutionState calls through the stale lambda into
    freed memory (SIGSEGV at ``stream_executor::DeviceAddressHandle::~``).
    See ``pymdp/ffi/__init__.py`` docstring for the full crash chain.

    Eagerly registering at module-import time lets the TypeRegistry settle
    *before* Mosaic-GPU forms any caches, so when those caches do form (at
    first JIT compile) the references they capture stay valid for the
    process lifetime. No-op when the FFI lib is absent or built without
    CUDA (CPU target alone always registers).
    """
    if not is_available():
        return
    # CPU fallback target — works on every platform, used when use_cuda is
    # False (e.g., PYMDP_FFI_USE_CUDA unset).
    _register("pymdp_neg_efe_all_policies", platform="cpu")
    caps = ffi_capabilities()
    if not caps.get("cuda", False):
        return
    # `JAX_PLATFORMS=cpu` (or any value omitting cuda/gpu) tells JAX not to
    # initialize the CUDA platform table at all. Registering FFI handlers
    # at platform="CUDA" against a missing platform leaves orphaned entries
    # that the subsequent CPU backend init rejects with the same
    # "Types used by FFI handlers must be registered before the handler
    # registration" precondition we documented below. Skip every CUDA-keyed
    # registration in that case — including the cuda_host stage that's
    # nominally platform="cpu", since it shares the NegEfeState type with
    # cuda_dev and the historical hazard is the TypeRegistry being only
    # half-populated. CPU-only paths still go through the plain
    # `pymdp_neg_efe_all_policies` target registered above.
    if not cuda_platform_allowed_by_env():
        return
    # CUDA-host multi-stage target (platform="cpu") and CUDA-dev multi-stage
    # target (platform="CUDA"). We intentionally do NOT gate on
    # has_jax_cuda_backend() here: that helper calls jax.devices() which
    # forces backend init, and on Tegra that init runs *after* the
    # cuda_host multi-stage registration has happened but *before* the
    # cuda_dev one — at which point XLA's CPU backend trips its
    # "Types used by FFI handlers must be registered before the handler
    # registration" precondition and the whole jax.devices() call raises,
    # leaving the cuda_dev target unregistered. The CUDA target registry is
    # platform-keyed in JAX, so registering on a host without a CUDA
    # backend is just a no-op entry in the table — it never gets resolved
    # by a non-CUDA JIT. Registering unconditionally also keeps the
    # TypeRegistry stable: by the time any backend init runs, every
    # platform variant we'll ever use is already in the table.
    _register_neg_efe_state_multistage(
        target_name="pymdp_neg_efe_all_policies_cuda_host",
        instantiate_symbol="pymdp_neg_efe_all_policies_cuda_host_instantiate",
        execute_symbol="pymdp_neg_efe_all_policies_cuda_host",
        platform="cpu",
    )
    _register_neg_efe_state_multistage(
        target_name="pymdp_neg_efe_all_policies_cuda_dev",
        instantiate_symbol="pymdp_neg_efe_cuda_dev_instantiate",
        execute_symbol="pymdp_neg_efe_cuda_dev_execute",
        platform="CUDA",
    )


@dataclass(frozen=True)
class RooflineParams:
    """Effective device constants for the neg-EFE CUDA routing model.

    All values are *effective* (sustained, not peak), measured on Orin (Ampere
    sm_87). Throughputs are flop/ms, bandwidths byte/ms, latencies ms. They are
    deliberately coarse: the routing decision is a comparison of two times, so
    it is robust to ~2x error in any single constant. Recalibrate per arch by
    editing `_ORIN_ROOFLINE` (or pass a `RooflineParams` to the model fn).
    """

    f_gpu: float  # CUDA-kernel f32 throughput (cuBLAS-staged contractions)
    bw_gpu: float  # CUDA-path device bandwidth
    launch_base_ms: float  # fixed CUDA dispatch + scatter overhead
    launch_per_stage_ms: float  # added latency per (modality | factor) stage
    f_xla: float  # fused-vmap throughput (XLA's generic fusion)
    bw_xla: float  # fused-vmap bandwidth
    fixed_xla_ms: float  # fused-vmap fixed dispatch


# Fit to an Orin bench sweep: the inductive winner (1.29e9 flop, M+F=6) lands at
# t_cuda~1.66 / t_jax~4.4 ms, the dense crossover at P~1-2k, and the many-factor
# regime (M+F up to 112) stays launch-bound. f_gpu>f_xla because the hand-tuned
# cuBLAS-staged kernel out-throughputs XLA's generic fusion on dense
# contractions; that win is what the per-stage launch term has to amortize.
_ORIN_ROOFLINE = RooflineParams(
    f_gpu=1.3e9,
    bw_gpu=1.0e8,
    launch_base_ms=0.5,
    launch_per_stage_ms=0.03,
    f_xla=3.4e8,
    bw_xla=1.0e8,
    fixed_xla_ms=0.6,
)


def neg_efe_ffi_beneficial(
    policy_matrix,
    qs_init: Sequence[jax.Array],
    A: Sequence[jax.Array],
    A_dependencies: Sequence[Sequence[int]],
    B_dependencies: Sequence[Sequence[int]],
    *,
    params: RooflineParams = _ORIN_ROOFLINE,
) -> bool | None:
    """Predict whether the CUDA kernel beats the fused JAX vmap for this shape.

    Routes by comparing two modelled wall times rather than thresholding one
    work scalar. The multi-stage kernel issues a launch per modality (A-scoring)
    and per factor (B-rollout) plus a fixed scatter; the fused vmap issues one.
    Each side's compute is a roofline — ``max(flops / throughput, bytes /
    bandwidth)`` — so the memory arm guards the low-arithmetic-intensity regime
    and the compute arm the dense one::

        t_cuda = launch_base + launch_per_stage*(M+F) + max(flops/f_gpu, bytes/bw_gpu)
        t_jax  = fixed_xla                            + max(flops/f_xla, bytes/bw_xla)

    Returns ``t_cuda < t_jax``, or ``None`` for degenerate shapes (caller then
    leaves the routing decision unconstrained). All inputs are static shapes,
    read via ``np.shape`` — no host<->device copy.

    Modelling choices: A[m] / B[f,u] are model params reused across the P
    policies, so their read volume is counted once; only the per-policy
    marginals/outputs (O_m, S_f) stream P*T times. The contraction volume per
    dependency group is a *product* over its parents — fan-out scales cost
    multiplicatively, which a per-factor sum does not capture.
    """
    pm_shape = np.shape(policy_matrix)
    if len(pm_shape) < 2 or not qs_init:
        return None
    P, T = int(pm_shape[0]), int(pm_shape[1])
    if P <= 0 or T <= 0:
        return None
    S = [int(np.shape(q)[0]) for q in qs_init]
    M, F = len(A_dependencies), len(B_dependencies)
    if M < 1 or F < 1:
        return None

    a_vol = 0  # sum_m O_m * prod S over A-deps  (A read volume / step)
    out_a = 0  # sum_m O_m                       (qo marginals / step)
    for m, deps in enumerate(A_dependencies):
        O_m = int(np.shape(A[m])[0])
        out_a += O_m
        vol = O_m
        for d in deps:
            vol *= S[d]
        a_vol += vol

    b_vol = 0  # sum_f S_f * prod S over B-deps  (B read volume / step)
    for f, deps in enumerate(B_dependencies):
        vol = S[f]
        for d in deps:
            vol *= S[d]
        b_vol += vol

    DT = 4  # f32 modelling width
    contraction = a_vol + b_vol
    flops = 2.0 * P * T * contraction
    # params (A, B) read once + per-policy marginals (O_m) and updated states (S_f)
    stream_elems = out_a + sum(S)
    bytes_moved = DT * (contraction + P * T * stream_elems)

    p = params
    t_cuda = (
        p.launch_base_ms
        + p.launch_per_stage_ms * (M + F)
        + max(flops / p.f_gpu, bytes_moved / p.bw_gpu)
    )
    t_jax = p.fixed_xla_ms + max(flops / p.f_xla, bytes_moved / p.bw_xla)
    return t_cuda < t_jax


def _cuda_routing_override() -> bool | None:
    """Manual bypass of the roofline routing model via ``PYMDP_FFI_CUDA_FORCE``.

    Returns True to force the CUDA target, False to force the fallback, or None
    to defer to the cost model.
    """
    force = os.environ.get("PYMDP_FFI_CUDA_FORCE", "").strip().lower()
    if force in ("1", "true", "yes"):
        return True
    if force in ("0", "false", "no"):
        return False
    return None


def _use_cuda_target(
    B_dependencies: Sequence[Sequence[int]],
    *,
    ffi_beneficial: bool | None = None,
) -> bool:
    """Predicate: should the CUDA FFI target be used over the CPU one?

    True when all hold:
      * `PYMDP_FFI_USE_CUDA` env var is set to a truthy value (opt-in),
      * the loaded lib was built with CUDA support (`ffi_capabilities()["cuda"]`),
      * every factor's B_dependencies rank is within the kernel's parent-rank cap,
      * the roofline model predicts CUDA is faster (``ffi_beneficial`` is not
        False), unless overridden via `PYMDP_FFI_CUDA_FORCE`.

    When the model declines, the caller falls through — to CPU FFI on a
    CPU-backend host (CPU targets register with platform="cpu" and are
    reachable), or to the JAX vmap on a CUDA-backend host (CPU targets are
    invisible to the CUDA jit there).
    """
    flag = os.environ.get("PYMDP_FFI_USE_CUDA", "").strip().lower()
    if flag in ("", "0", "false", "no"):
        return False
    if not ffi_capabilities().get("cuda"):
        return False
    if not all(dep_list_rank_ok(d) for d in B_dependencies):
        return False
    override = _cuda_routing_override()
    if override is not None:
        return override
    if ffi_beneficial is False:
        return False
    return True


def can_handle(
    A_dependencies: Sequence[Sequence[int]],
    B_dependencies: Sequence[Sequence[int]],
    use_param_info_gain: bool,
    *,
    num_factors: int | None = None,
    num_modalities: int | None = None,
    ffi_beneficial: bool | None = None,
) -> bool:
    """Fast predicate: can the FFI kernel handle this problem?

    Restrictions:
      * Each modality's A_dependencies rank must be in [1, 8].
      * Each factor's B_dependencies rank must be in [1, 8].
      * When num_factors/num_modalities are provided, dependency list lengths
        and factor indices must match the model metadata.

    Note on B_dependencies: general (non-factor-local) B-deps are supported via
    per-factor Kronecker B-propagation in the kernel; FactorMeta encodes the
    parent topology so factor-local and multi-parent shapes share one path.
    The factor-local restriction in the inductive perf fixtures comes from
    `generate_I_matrix` (per-factor backward reachability), not from this gate.

    Note on use_param_info_gain: supported on CPU and CUDA; the kernels compute
    the pA/pB novelty (Dirichlet information-gain) terms via precomputed wA /
    wB weights derived from the spm_wnorm formula.

    Note on ffi_beneficial: callers on a CUDA-backend host can pass the
    roofline routing decision (see ``neg_efe_ffi_beneficial``); when it is
    False the gate declines so the JAX vmap handles the call (CPU FFI is
    invisible to the CUDA jit). Omitting it preserves prior behavior (FFI
    always wins on CPU-backend hosts).
    """
    if not is_available():
        return False
    # CPU FFI targets are registered with platform="cpu" and are invisible to
    # the CUDA JIT.  On a CUDA-backend host, only allow dispatch when the CUDA
    # kernel path is actually selected; otherwise JAX handles the call.
    if has_jax_cuda_backend():
        cuda_selected = _use_cuda_target(
            B_dependencies, ffi_beneficial=ffi_beneficial
        )
        if not cuda_selected:
            return False
        # The CUDA modality kernel only covers rank<=3; rank>=4 A-deps must fall
        # back to JAX rather than dispatch into a missing kernel (hard error).
        if any(len(deps) > CUDA_MODALITY_MAX_RANK for deps in A_dependencies):
            return False
    if num_modalities is not None and len(A_dependencies) != num_modalities:
        return False
    if num_factors is not None and len(B_dependencies) != num_factors:
        return False
    for deps in A_dependencies:
        if not dep_list_rank_ok(deps):
            return False
        if num_factors is not None and not dep_indices_in_range(deps, num_factors):
            return False
    for deps in B_dependencies:
        if not dep_list_rank_ok(deps):
            return False
        if num_factors is not None and not dep_indices_in_range(deps, num_factors):
            return False
    return True


def _validate_efe_inputs(
    policy_matrix: jax.Array,
    qs_init: Sequence[jax.Array],
    A: Sequence[jax.Array],
    B: Sequence[jax.Array],
    C: Sequence[jax.Array],
    I: Sequence[jax.Array],
    pA: Sequence[jax.Array] | None,
    pB: Sequence[jax.Array] | None,
    A_dependencies: Sequence[Sequence[int]],
    B_dependencies: Sequence[Sequence[int]],
    *,
    use_inductive: bool,
    use_param_info_gain: bool,
    inductive_epsilon: float,
    ffi_beneficial: bool | None = None,
) -> None:
    F = len(qs_init)
    M = len(A)
    if F < 1 or M < 1:
        raise ValueError("neg-EFE FFI requires at least one factor and one modality")
    if len(B) != F or len(I) != F:
        raise ValueError("B and I must have one entry per hidden-state factor")
    if len(C) != M:
        raise ValueError("C must have one entry per observation modality")
    # ffi_beneficial is threaded in so the gate's decision matches the
    # dispatch path below: _use_cuda_target consults it to skip CUDA on shapes
    # the roofline model predicts are launch-bound (CPU FFI targets are
    # invisible to the CUDA jit), and can_handle must use the same input or the
    # validator accepts calls that later fail at FFI lookup time.
    if not can_handle(
        A_dependencies,
        B_dependencies,
        use_param_info_gain=use_param_info_gain,
        num_factors=F,
        num_modalities=M,
        ffi_beneficial=ffi_beneficial,
    ):
        raise ValueError("unsupported or inconsistent A/B dependency metadata")

    if use_param_info_gain:
        if (pA is None or len(pA) == 0) and (pB is None or len(pB) == 0):
            raise ValueError("use_param_info_gain=True requires at least one of pA, pB")
        if pA is not None and len(pA) > 0:
            if len(pA) != M:
                raise ValueError(
                    f"pA must have one entry per modality (M={M}); got {len(pA)}"
                )
            for m, (pa_m, a_m) in enumerate(zip(pA, A)):
                if np.shape(pa_m) != np.shape(a_m):
                    raise ValueError(
                        f"pA[{m}] shape {np.shape(pa_m)} must match A[{m}] shape "
                        f"{np.shape(a_m)}"
                    )
        if pB is not None and len(pB) > 0:
            if len(pB) != F:
                raise ValueError(
                    f"pB must have one entry per factor (F={F}); got {len(pB)}"
                )
            for f, (pb_f, b_f) in enumerate(zip(pB, B)):
                if np.shape(pb_f) != np.shape(b_f):
                    raise ValueError(
                        f"pB[{f}] shape {np.shape(pb_f)} must match B[{f}] shape "
                        f"{np.shape(b_f)}"
                    )

    # np.shape(x) reads .shape if present and falls back to constructing a
    # numpy array only for plain Python lists — never forces a JAX-array
    # conversion or host↔device copy. Validation runs at trace time once per
    # jit cache miss, so the savings are small but the intent is clearer:
    # we're inspecting metadata, not materializing data.
    pm_shape = np.shape(policy_matrix)
    if len(pm_shape) != 3:
        raise ValueError(f"policy_matrix must have shape (P, T, F); got {pm_shape}")
    P, T, F_policy = pm_shape
    if P <= 0 or T <= 0 or F_policy != F:
        raise ValueError(
            f"policy_matrix shape {pm_shape} is inconsistent with {F} factors"
        )

    qs_shapes = [np.shape(q) for q in qs_init]
    S = [int(s[0]) for s in qs_shapes]
    for f, shape in enumerate(qs_shapes):
        if len(shape) != 1 or S[f] <= 0:
            raise ValueError(f"qs_init[{f}] must have shape (S_f,), got {shape}")

    A_shapes = [np.shape(a_m) for a_m in A]
    for m, (shape, deps) in enumerate(zip(A_shapes, A_dependencies)):
        expected = (int(shape[0]),) + tuple(S[d] for d in deps)
        if shape != expected:
            raise ValueError(f"A[{m}] shape {shape} does not match expected {expected}")
    for f, (b_f, deps) in enumerate(zip(B, B_dependencies)):
        shape = np.shape(b_f)
        expected = (S[f],) + tuple(S[d] for d in deps) + (int(shape[-1]),)
        if shape != expected or shape[-1] <= 0:
            raise ValueError(f"B[{f}] shape {shape} does not match expected {expected}")
    for m, c_m in enumerate(C):
        shape = np.shape(c_m)
        O_m = A_shapes[m][0]
        if len(shape) == 1:
            if shape[0] != O_m:
                raise ValueError(
                    f"C[{m}] must have shape (O_m,) or (T, O_m); got {shape}"
                )
        elif len(shape) == 2:
            if shape[0] != T or shape[1] != O_m:
                raise ValueError(
                    f"C[{m}] must have shape (O_m,) or (T, O_m); got {shape}"
                )
        else:
            raise ValueError(f"C[{m}] must have shape (O_m,) or (T, O_m); got {shape}")
    for f, i_f in enumerate(I):
        shape = np.shape(i_f)
        if len(shape) != 2 or shape[0] <= 0 or shape[1] != S[f]:
            raise ValueError(f"I[{f}] must have shape (depth_f, S_f); got {shape}")

    if use_inductive:
        # Skip the bound check when epsilon is a tracer (jit/vmap); the kernel
        # re-validates at runtime per batch element. float() on a tracer raises
        # TypeError (TracerArrayConversionError is a TypeError subclass).
        try:
            eps = float(inductive_epsilon)
        except TypeError:
            return
        if eps <= 0.0:
            raise ValueError(
                "inductive_epsilon must be positive when use_inductive=True"
            )


def neg_efe_all_policies_ffi(
    policy_matrix: jax.Array,  # [P, T, F], int32
    qs_init: Sequence[jax.Array],  # list of F, each shape (S_f,)
    A: Sequence[jax.Array],  # list of M, each shape (O_m, S_{dep0}, ...)
    B: Sequence[jax.Array],  # list of F, each shape (S_f, S_{B_dep0}, ..., U_f)
    C: Sequence[jax.Array],  # list of M, each shape (O_m,) or (T, O_m)
    I: Sequence[jax.Array],  # list of F, each shape (depth_f, S_f)
    A_dependencies: Sequence[Sequence[int]],
    B_dependencies: Sequence[Sequence[int]],
    *,
    pA: Sequence[jax.Array]
    | None = None,  # list of M, same shape as A; required when use_param_info_gain
    pB: Sequence[jax.Array]
    | None = None,  # list of F, same shape as B; required when use_param_info_gain
    use_utility: bool = True,
    use_states_info_gain: bool = True,
    use_param_info_gain: bool = False,
    use_inductive: bool = True,
    inductive_epsilon: float = 1e-3,
    ffi_cache_params_static: bool = False,
) -> jax.Array:
    """Fused neg-EFE kernel — production path for `update_posterior_policies_inductive`.

    Returns `neg_efe` of shape (P,). Caller applies gamma + softmax + log E.
    `C` may be time-independent `(O_m,)`; the wrapper broadcasts it to the
    kernel's `(T, O_m)` ABI. Utility / state info gain / inductive / param
    info gain terms are toggled by keyword flags. When use_param_info_gain is
    on, pass `pA` and/or `pB` (Dirichlet posterior parameters with the same
    shape as A/B); the kernel computes the wA / wB-tr novelty weights once per
    call and folds the per-(node, modality) and per-(node, factor) terms into
    the policy score.
    """
    # The CUDA gate verifies the kernel was built into the lib + falls through
    # to the CPU target otherwise. CPU and CUDA can ship in one dylib; runtime
    # env picks whether CUDA is preferred.
    #
    # When jaxlib actually has a CUDA backend (desktop with NVIDIA GPU +
    # jax[cuda]), we additionally register the platform="CUDA" device-buffer
    # target (NegEfeCudaDev) and dispatch via lax.platform_dependent so the
    # JIT compiles in the device-pointer path on CUDA and the host-buffer
    # path on CPU. On Jetson Nano (CPU-only jaxlib), has_jax_cuda_backend()
    # is False and only the host-buffer target is registered.
    # Run the roofline routing model on concrete shapes so the CUDA gate here
    # matches the decision can_handle made upstream (batch_size is ignored;
    # vmap_method="broadcast_all" handles it internally and per-call dispatch
    # overhead is the same).
    _cuda_ok = neg_efe_ffi_beneficial(
        policy_matrix, qs_init, A, A_dependencies, B_dependencies
    )
    use_cuda = _use_cuda_target(B_dependencies, ffi_beneficial=_cuda_ok)
    use_cuda_dev = use_cuda and has_jax_cuda_backend()

    if use_cuda:
        # Host-buffer CUDA target — multi-stage on platform="cpu", same
        # per-JIT NegEfeState ownership as the device-buffer dev target. The
        # two registrations bind to the same NegEfeState type.
        target_name = "pymdp_neg_efe_all_policies_cuda_host"
        _register_neg_efe_state_multistage(
            target_name=target_name,
            instantiate_symbol="pymdp_neg_efe_all_policies_cuda_host_instantiate",
            execute_symbol=target_name,
            platform="cpu",
        )
    else:
        target_name = "pymdp_neg_efe_all_policies"
        _register(target_name)
    if use_cuda_dev:
        _register_neg_efe_state_multistage(
            target_name="pymdp_neg_efe_all_policies_cuda_dev",
            instantiate_symbol="pymdp_neg_efe_cuda_dev_instantiate",
            execute_symbol="pymdp_neg_efe_cuda_dev_execute",
            platform="CUDA",
        )

    _validate_efe_inputs(
        policy_matrix,
        qs_init,
        A,
        B,
        C,
        I,
        pA,
        pB,
        A_dependencies,
        B_dependencies,
        use_inductive=use_inductive,
        use_param_info_gain=use_param_info_gain,
        inductive_epsilon=inductive_epsilon,
        ffi_beneficial=_cuda_ok,
    )

    # C is broadcast to the kernel's `(T, O_m)` ABI; shape attrs are int64.
    S = np.array([int(q.shape[0]) for q in qs_init], dtype=np.int64)
    O = np.array([int(a.shape[0]) for a in A], dtype=np.int64)
    U = np.array([int(b.shape[-1]) for b in B], dtype=np.int64)
    I_depths = np.array([int(i_arr.shape[0]) for i_arr in I], dtype=np.int64)

    T = int(np.shape(policy_matrix)[1])
    # `jnp.broadcast_to` accepts numpy/jax inputs and returns a jax array;
    # `np.ndim` reads metadata without forcing a materialization.
    C_for_ffi = [
        jnp.broadcast_to(c_m, (T, int(O[m]))) if np.ndim(c_m) == 1 else c_m
        for m, c_m in enumerate(C)
    ]

    qs_flat, qs_offsets = flatten_ragged_f32(qs_init)
    A_flat, A_offsets = flatten_ragged_f32(A)
    B_flat, B_offsets = flatten_ragged_f32(B)
    C_flat, C_offsets = flatten_ragged_f32(C_for_ffi)
    I_flat, I_offsets = flatten_ragged_f32(I)

    # pA / pB: when present, share strides with A / B (same shapes); when
    # absent, pass a zero-element buffer so the kernel sees `element_count==0`
    # and skips the corresponding precompute. The kernel still requires both
    # buffers in the call signature regardless of the flag.
    if use_param_info_gain and pA is not None and len(pA) > 0:
        pA_flat, _ = flatten_ragged_f32(pA)
    else:
        pA_flat = jnp.zeros((0,), dtype=jnp.float32)
    if use_param_info_gain and pB is not None and len(pB) > 0:
        pB_flat, _ = flatten_ragged_f32(pB)
    else:
        pB_flat = jnp.zeros((0,), dtype=jnp.float32)

    A_dep_flat, A_dep_offsets = flatten_dep_lists(A_dependencies)
    B_dep_flat, B_dep_offsets = flatten_dep_lists(B_dependencies)

    flags = 0
    if use_utility:
        flags |= FLAG_USE_UTILITY
    if use_states_info_gain:
        flags |= FLAG_USE_STATES_INFO_GAIN
    if use_inductive:
        flags |= FLAG_USE_INDUCTIVE
    if use_param_info_gain:
        flags |= FLAG_USE_PARAM_INFO_GAIN
    # Warm-check hint for the CUDA-Dev path (no effect on CPU/host): set → params
    # static across calls → devptr fast warm check; clear (default) → rebuild the
    # caches from the device buffers every call (always correct, the learning regime).
    # The FLAG_MODEL_PARAMS_STATIC bit name mirrors the C++ ABI (kFlagModelParamsStatic);
    # the Python-facing knob is `ffi_cache_params_static`. See
    # docs/ffi_cpu_cache_static_hint_rejected.md for why this is CUDA-only.
    if ffi_cache_params_static:
        flags |= FLAG_MODEL_PARAMS_STATIC

    P = policy_matrix.shape[0]
    out_type = jax.ShapeDtypeStruct((P,), jnp.float32)

    # vmap_method="broadcast_all": under vmap (Agent.infer_policies' batch),
    # JAX prepends a leading batch dim to every input buffer and the output,
    # so we get one FFI dispatch for the whole batch instead of Bn sequential
    # ones. The kernel detects the batched form from the rank of
    # policy_matrix and iterates internally; ~50us of XLA dispatch overhead
    # per batch element collapses into a single call.
    # epsilon is a 0-d float32 buffer (runtime input), so it can be a vmap
    # tracer; broadcast_all presents it to the kernel as [B] under vmap.
    eps_buf = jnp.asarray(inductive_epsilon, dtype=jnp.float32).reshape(())

    pm_i32 = policy_matrix.astype(jnp.int32)
    static_attrs = dict(
        S=S,
        O=O,
        U=U,
        qs_offsets=qs_offsets,
        A_offsets=A_offsets,
        B_offsets=B_offsets,
        C_offsets=C_offsets,
        I_offsets=I_offsets,
        I_depths=I_depths,
        A_dep_flat=A_dep_flat,
        A_dep_offsets=A_dep_offsets,
        B_dep_flat=B_dep_flat,
        B_dep_offsets=B_dep_offsets,
        flags=np.int32(flags),
    )

    def _make_call(name: str):
        return jax.ffi.ffi_call(name, out_type, vmap_method="broadcast_all")

    if use_cuda_dev:
        # lax.platform_dependent compiles in the device-buffer target when
        # the JIT lowers for CUDA, and the host-buffer target when the JIT
        # lowers for CPU. Both branches share the buffer prep above; only
        # the FFI target name differs. Static attrs are captured by closure
        # so the branch signatures match what platform_dependent expects.
        cpu_call_host = _make_call("pymdp_neg_efe_all_policies_cuda_host")
        cuda_call_dev = _make_call("pymdp_neg_efe_all_policies_cuda_dev")

        def _cpu_branch(pm, qs, A, B, C, I, pA, pB, eps):
            return cpu_call_host(pm, qs, A, B, C, I, pA, pB, eps, **static_attrs)

        def _cuda_branch(pm, qs, A, B, C, I, pA, pB, eps):
            return cuda_call_dev(pm, qs, A, B, C, I, pA, pB, eps, **static_attrs)

        return jax.lax.platform_dependent(
            pm_i32,
            qs_flat,
            A_flat,
            B_flat,
            C_flat,
            I_flat,
            pA_flat,
            pB_flat,
            eps_buf,
            cpu=_cpu_branch,
            cuda=_cuda_branch,
        )

    call = _make_call(target_name)
    return call(
        pm_i32,
        qs_flat,
        A_flat,
        B_flat,
        C_flat,
        I_flat,
        pA_flat,
        pB_flat,
        eps_buf,
        **static_attrs,
    )


__all__ = [
    "FLAG_USE_UTILITY",
    "FLAG_USE_STATES_INFO_GAIN",
    "FLAG_USE_INDUCTIVE",
    "FLAG_USE_PARAM_INFO_GAIN",
    "FLAG_MODEL_PARAMS_STATIC",
    "can_handle",
    "neg_efe_all_policies_ffi",
]
