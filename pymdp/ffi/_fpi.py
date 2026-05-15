"""Python wrapper for the generalized fused-FPI FFI kernel."""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from ._core import (
    _register,
    dep_indices_in_range,
    dep_list_rank_ok,
    flatten_dep_lists,
    has_jax_cuda_backend,
    is_available,
)


def can_handle_fpi(
    prior: Sequence[jax.Array],
    A_dependencies: Sequence[Sequence[int]],
    num_iter: int,
) -> bool:
    """True iff the generalized FPI kernel can dispatch this problem.

    Restrictions:
      * num_iter must be > 0
      * len(prior) >= 1
      * each modality's A_dependencies rank in [1, 8]
      * library must be loadable

    K=1/K=2/K=3 use specialized hot paths in the kernel; K>=4 dispatches to
    a generic forward-chain path (modality_Kn) — same auto-vectorized inner
    primitives, runtime K. Upper bound is kMaxFfiDependencyRank in
    pymdp/ffi/src/neg_efe_layout.h (= MAX_FFI_DEP_RANK in _core.py).
    """
    if not is_available():
        return False
    # On a CUDA-backend host the pymdp_fpi_cuda_host shim (registered below)
    # D2H/H2D-wraps the CPU kernel so the same gate covers both platforms.
    # JAX's CPU backend uses pymdp_fpi directly; lax.platform_dependent in
    # fpi_ffi() picks the right target at JIT time.
    if num_iter <= 0:
        return False
    F = len(prior)
    if F < 1:
        return False
    M = len(A_dependencies)
    if M < 1:
        return False
    for deps in A_dependencies:
        if not dep_list_rank_ok(deps):
            return False
        if not dep_indices_in_range(deps, F):
            return False
        # Distinct factors per modality is a contract that the kernel relies on
        # for __restrict__ correctness on per-factor q / log_q slices. Duplicate
        # deps would alias `q[deps[i]] == q[deps[j]]` and break the no-alias
        # assumption. Reject the gate; FPI falls back to the JAX scan path.
        if len(set(deps)) != len(deps):
            return False
    return True


def _native_cuda_fpi_eligible(A_dependencies: Sequence[Sequence[int]]) -> bool:
    """True iff every modality's A_dependencies rank is in [1, 3].

    The native CUDA FPI kernel (pymdp_fpi_cuda_native) only implements K<=3
    modality marginals — K>=4 dispatches the kernel's switch default which
    silently produces no work. The host gate filters those cases here so the
    K>=4 path routes through the pymdp_fpi_cuda_host shim instead. Every
    production rollout fixture (rollout_loop / rollout_realistic /
    rollout_learning) lands in K<=3 so this gate fires for the cases that
    matter most. fpi_high_rank (K=5) takes the shim.

    Note: a previous commit (49a4265, reverted) tried to also gate on
    "no batch dim" via np.ndim(prior[0]) < 2, intending to route batch=1
    standalone calls (fpi_large) to the shim. That broke the rollouts —
    under vmap_method="broadcast_all" the inner trace sees the unbatched
    shape regardless of whether vmap will be applied at runtime, so the
    check couldn't distinguish standalone batch=1 from rollouts. The
    proper fix for fpi_large's under-utilization is the cooperative-grid
    multi-block-per-batch kernel sketched in jetson-orin-fpi-idea.md;
    accept the standalone-batch=1 regression in the meantime.
    """
    for deps in A_dependencies:
        if len(deps) > 3:
            return False
    return True


def _validate_fpi_inputs(
    log_likelihoods: Sequence[jax.Array],
    log_prior: Sequence[jax.Array],
    A_dependencies: Sequence[Sequence[int]],
    num_iter: int,
) -> None:
    F = len(log_prior)
    M = len(log_likelihoods)
    if F < 1 or M < 1:
        raise ValueError("FPI FFI requires at least one factor and one modality")
    if not can_handle_fpi(log_prior, A_dependencies, num_iter):
        raise ValueError(
            "FPI FFI cannot handle this problem (rank > 8, num_iter <= 0, or "
            "library unavailable)"
        )
    if len(A_dependencies) != M:
        raise ValueError(
            f"A_dependencies length {len(A_dependencies)} does not match "
            f"len(log_likelihoods) = {M}"
        )

    # np.shape reads metadata without forcing a JAX-array conversion. The
    # trailing axis is the per-factor (or per-modality kron) state dim; under
    # vmap a leading batch axis may be present and is handled by `_flat`.
    lp_shapes = [np.shape(lp) for lp in log_prior]
    S = [int(shape[-1]) for shape in lp_shapes]
    for f, shape in enumerate(lp_shapes):
        if len(shape) < 1 or S[f] <= 0:
            raise ValueError(f"log_prior[{f}] must have a trailing state axis; got {shape}")

    for m, (ll, deps) in enumerate(zip(log_likelihoods, A_dependencies)):
        ll_shape = np.shape(ll)
        # ll trailing dims are S[deps[0]], S[deps[1]], ...; leading dims (if
        # any) are the vmap batch — anything before the last len(deps) axes.
        rank = len(deps)
        if len(ll_shape) < rank:
            raise ValueError(
                f"log_likelihoods[{m}] rank {len(ll_shape)} < |A_deps[{m}]| = {rank}"
            )
        trailing = ll_shape[len(ll_shape) - rank:]
        expected = tuple(S[d] for d in deps)
        if trailing != expected:
            raise ValueError(
                f"log_likelihoods[{m}] trailing shape {trailing} does not match "
                f"expected {expected} from A_dependencies[{m}] = {tuple(deps)}"
            )


def fpi_ffi(
    log_likelihoods: Sequence[jax.Array],   # M tensors, ll_m shape S[A_deps[m]]
    log_prior: Sequence[jax.Array],         # F tensors, log_prior_f shape (S_f,)
    A_dependencies: Sequence[Sequence[int]],
    num_iter: int,
) -> list[jax.Array]:
    """Generalized fused-FPI kernel.

    Returns the post-softmax marginal posteriors `qs` (length-F list, qs[f] of
    shape (S_f,)). Internally runs up to `num_iter` fixed-point iterations in
    C++ to collapse the per-iteration JAX kernel-launch overhead.

    Under `vmap` (Agent's batch dim) the kernel detects the batch from a
    leading axis on `lp_flat` and iterates per-batch internally — single XLA
    dispatch per call.

    Early-stop: kernel checks `max|log_q[k] - log_q[k-1]|` after each body
    iteration and breaks below a hard-coded 1e-5 tolerance — well under
    test_fpi_ffi.py's parity atol of 1e-6 (max observed |q - q_ref| was
    2.7e-7), still loose enough to fire on small/easy shapes (fpi_inference,
    fpi_high_rank) where 16 iters typically over-converges. Larger fixtures
    (fpi_large) don't reach the threshold within num_iter=16 and run the
    full loop. The per-iter snapshot that feeds the check is folded into
    the softmax pass on both CPU and CUDA (single load of log_q feeds both
    softmax and the snapshot write), so the always-on check costs only one
    max-abs-diff (CPU) / one block_reduce_max (CUDA) per iter — small enough
    that the early-exit benefit on convergent shapes always outweighs the
    overhead on non-convergent shapes.

    The custom_vjp backward path uses the JAX scan reference (full num_iter),
    so gradients always reflect the unbounded forward.
    """
    _register("pymdp_fpi", platform="cpu")
    # On a CUDA-backend host, also register the platform="CUDA" targets:
    # the native CUDA FPI kernel (preferred — runs all num_iter iterations
    # on the GPU stream, no host roundtrip) and the host-buffer shim that
    # D2H/H2D-wraps the CPU kernel (fallback for K>=4 modalities). Pick
    # one or the other in lax.platform_dependent below based on the gate.
    # On CPU-only jaxlib this branch is skipped and the CPU target alone
    # serves both paths.
    use_cuda = has_jax_cuda_backend()
    if use_cuda:
        _register("pymdp_fpi_cuda_host", platform="CUDA")
        _register("pymdp_fpi_cuda_native", platform="CUDA")
    use_native_cuda = use_cuda and _native_cuda_fpi_eligible(A_dependencies)

    _validate_fpi_inputs(log_likelihoods, log_prior, A_dependencies, num_iter)

    F = len(log_prior)
    M = len(log_likelihoods)

    # S[f] = trailing-dim size; prior[f] has shape (S_f,) pre-vmap or
    # (batch, S_f) under vmap_method="broadcast_all" (which we use).
    S = np.array([int(p.shape[-1]) for p in log_prior], dtype=np.int64)

    # Per-modality offsets into the flat ll buffer (per-batch element).
    ll_offsets = np.zeros(M + 1, dtype=np.int64)
    for m, deps in enumerate(A_dependencies):
        ll_offsets[m + 1] = ll_offsets[m] + int(np.prod([S[d] for d in deps]))

    # Per-factor offsets (also used for the output q_flat layout).
    lp_offsets = np.zeros(F + 1, dtype=np.int64)
    for f in range(F):
        lp_offsets[f + 1] = lp_offsets[f] + int(S[f])

    A_dep_flat, A_dep_offsets = flatten_dep_lists(A_dependencies)

    # ll_m is shape (S[d0], S[d1], ...) pre-vmap or (batch, S[d0], ...) under
    # vmap_method="broadcast_all". Flatten the trailing state-factor dims into
    # one and concatenate along that axis — leading batch dim (if any) is
    # preserved.
    def _flat(a: jax.Array, ndim_state: int) -> jax.Array:
        if a.ndim == ndim_state:
            return a.reshape(-1)
        leading = a.shape[: a.ndim - ndim_state]
        return a.reshape(leading + (-1,))

    ll_flats = [_flat(ll, len(deps))
                for ll, deps in zip(log_likelihoods, A_dependencies)]
    lp_flats = [_flat(lp, 1) for lp in log_prior]
    ll_flat = jnp.concatenate(ll_flats, axis=-1)
    lp_flat = jnp.concatenate(lp_flats, axis=-1)

    total_S = int(lp_offsets[F])
    out_type = jax.ShapeDtypeStruct(lp_flat.shape, jnp.float32)

    # vmap_method="broadcast_all": single FFI dispatch per vmapped batch; the
    # kernel detects the batch from lp_flat.element_count() / total_S.
    static_attrs = dict(
        S=S,
        ll_offsets=ll_offsets,
        lp_offsets=lp_offsets,
        A_dep_flat=A_dep_flat,
        A_dep_offsets=A_dep_offsets,
        num_iter=np.int32(num_iter),
    )

    def _make_call(name: str):
        return jax.ffi.ffi_call(name, out_type, vmap_method="broadcast_all")

    if use_cuda:
        # lax.platform_dependent compiles in the CUDA target when the JIT
        # lowers for CUDA, and the CPU target otherwise. Same buffer prep
        # for both branches; only the target name differs. The CUDA branch
        # picks pymdp_fpi_cuda_native (native kernel, no host roundtrip)
        # when every modality's K is in [1, 3], else falls back to the
        # pymdp_fpi_cuda_host shim that wraps the CPU kernel.
        cpu_call = _make_call("pymdp_fpi")
        cuda_target = "pymdp_fpi_cuda_native" if use_native_cuda else "pymdp_fpi_cuda_host"
        cuda_call = _make_call(cuda_target)

        def _cpu_branch(ll, lp):
            return cpu_call(ll, lp, **static_attrs)

        def _cuda_branch(ll, lp):
            return cuda_call(ll, lp, **static_attrs)

        q_flat = jax.lax.platform_dependent(
            ll_flat, lp_flat, cpu=_cpu_branch, cuda=_cuda_branch,
        )
    else:
        q_flat = _make_call("pymdp_fpi")(ll_flat, lp_flat, **static_attrs)

    # Slice the flat output back into per-factor arrays. Under vmap the
    # leading batch axis is preserved by lax.dynamic_slice / fancy indexing
    # — we use ... indexing so the same code handles batched and unbatched.
    return [q_flat[..., int(lp_offsets[f]): int(lp_offsets[f + 1])]
            for f in range(F)]


def _register_all_fpi_targets_eager() -> None:
    """Register every FPI FFI target at import time.

    Mirror of ``_register_all_efe_targets_eager`` in ``_efe.py``. FPI is
    single-stage (no NegEfeState type), so this only goes through
    ``jax.ffi.register_ffi_target`` — but the same TypeRegistry-perturbation
    hazard applies via the multi-stage neg-EFE registrations that share the
    process. Doing all registrations eagerly, in one phase before any JIT
    trace, keeps the registry stable for the rest of the process lifetime.
    """
    if not is_available():
        return
    # Same reasoning as _register_all_efe_targets_eager: register on both
    # platforms unconditionally. has_jax_cuda_backend() forces backend init,
    # which trips an FFI ordering precondition on Tegra during eager mode;
    # the CUDA entry is a harmless no-op when no CUDA backend is present.
    _register("pymdp_fpi", platform="cpu")
    _register("pymdp_fpi_cuda_host", platform="CUDA")
    _register("pymdp_fpi_cuda_native", platform="CUDA")


__all__ = [
    "can_handle_fpi",
    "fpi_ffi",
]
