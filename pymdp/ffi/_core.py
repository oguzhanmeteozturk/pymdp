"""Shared library loading and target registration for pymdp FFI kernels."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path
from typing import Callable, Sequence, TypeVar

import jax
import jax.numpy as jnp
import numpy as np

T = TypeVar("T")

if sys.platform == "win32":
    _LIB_NAME = "pymdp_ffi.dll"
else:
    _LIB_NAME = {
        "darwin": "libpymdp_ffi.dylib",
        "linux": "libpymdp_ffi.so",
    }.get(sys.platform, "libpymdp_ffi.so")

_lib = None
_load_error: Exception | None = None
_registered_ffi_targets: set[str] = set()
_registered_ffi_types: set[str] = set()

# Must match PYMDP_FFI_CAP_* in src/cpu_capabilities.h.
_CAP_AARCH64_NEON = 1 << 0
_CAP_F16_FML = 1 << 1
_CAP_CUDA = 1 << 2
_CAP_X86_AVX2 = 1 << 3
_CAP_X86_AVX512F = 1 << 4

# Must match pymdp_ffi::kMaxFfiDependencyRank in pymdp/ffi/src/neg_efe_layout.h.
MAX_FFI_DEP_RANK = 8


def _library_path() -> Path:
    override = os.environ.get("PYMDP_FFI_LIB")
    if override:
        return Path(override)
    return Path(__file__).parent / "build" / _LIB_NAME


def _load_library() -> None:
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return
    path = _library_path()
    if not path.exists():
        # Without -DPython_EXECUTABLE the build hits jax.ffi.include_dir()
        # via system python and fails with ModuleNotFoundError.
        _load_error = FileNotFoundError(
            f"pymdp FFI library not found at {path}. "
            + "Build with: cmake -S pymdp/ffi -B pymdp/ffi/build -G 'Unix Makefiles' "
            + "-DPython_EXECUTABLE=$(which python) "
            + "&& cmake --build pymdp/ffi/build"
        )
        return
    try:
        _lib = ctypes.CDLL(str(path))
    except OSError as exc:
        _load_error = exc


def _require_loaded_lib(*, registering_target: str | None = None) -> ctypes.CDLL:
    """Return the ctypes handle or raise with the stored load error."""
    _load_library()
    if _lib is None:
        if registering_target is not None:
            raise RuntimeError(
                f"Cannot register FFI target {registering_target!r}: {_load_error}"
            )
        raise RuntimeError(f"pymdp FFI library unavailable: {_load_error}")
    return _lib


def dep_list_rank_ok(deps: Sequence[int]) -> bool:
    """True if ``deps`` length is within [1, MAX_FFI_DEP_RANK] (kernel contract)."""
    n = len(deps)
    return 1 <= n <= MAX_FFI_DEP_RANK


def dep_indices_in_range(deps: Sequence[int], num_factors: int) -> bool:
    """True if every factor index is in ``[0, num_factors)``."""
    return all(0 <= d < num_factors for d in deps)


def _register(target_name: str, symbol_name: str | None = None, platform: str = "cpu") -> None:
    # Register key is (target_name, platform) so a target available on both
    # CPU and CUDA can be registered twice with different symbols pointing
    # at distinct C entry points.
    key = f"{target_name}@{platform}"
    if key in _registered_ffi_targets:
        return
    lib = _require_loaded_lib(registering_target=target_name)
    symbol = getattr(lib, symbol_name or target_name)  # pyright: ignore[reportAny]
    jax.ffi.register_ffi_target(
        target_name,
        jax.ffi.pycapsule(symbol),  # pyright: ignore[reportUnknownMemberType,reportAny]
        platform=platform,
    )
    _registered_ffi_targets.add(key)


def _data_pycapsule(addr: int) -> object:
    """Wrap a raw void* address in a PyCapsule with no destructor.

    jax.ffi.pycapsule wraps a ctypes function pointer (callable address);
    register_ffi_type wants capsules around DATA pointers (the XLA_FFI_TypeId
    and XLA_FFI_TypeInfo structs themselves). XLA dereferences the capsule's
    pointer as the typed struct — it never calls it as a function. The
    destructor is null because the type-id / type-info storage is a static
    global on the C++ side and outlives the capsule.
    """
    builder = ctypes.pythonapi.PyCapsule_New
    builder.restype = ctypes.py_object
    builder.argtypes = (ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p)
    return builder(ctypes.c_void_p(addr), None, None)


def has_jax_cuda_backend() -> bool:
    """True when **some** visible JAX device reports platform ``gpu``/``cuda``.

    This does **not** mean the default JIT backend is GPU — only that a CUDA
    device exists so registering the platform=\"CUDA\" NegEfeCudaDev target is
    meaningful. Jetson Nano with CPU-only jaxlib returns False (use host-buffer
    CUDA FFI). Desktop without NVIDIA hardware typically returns False too.
    """
    try:
        return any(d.platform == "gpu" or d.platform == "cuda" for d in jax.devices())
    except RuntimeError:
        # jax.devices() can raise when no backend is available for the
        # default platform. Treat as "no CUDA backend".
        return False


def _register_multistage(
    target_name: str,
    instantiate_symbol: str,
    execute_symbol: str,
    type_name: str,
    type_id_symbol: str,
    type_info_symbol: str,
    platform: str,
) -> None:
    """Register a multi-stage FFI target with State<T> on the given platform.

    Registers (a) the XLA FFI type for the per-instance state, then (b) the
    target itself as a {"instantiate", "execute"} stage dict. Idempotent —
    repeat calls for the same target_name are a no-op.

    Used by both the platform="CUDA" NegEfeCudaDev target and the
    platform="cpu" NegEfeCudaHost target — same multi-stage shape, just
    different XLA platform tags so JAX dispatches the right one.

    Caller is responsible for any per-platform gating (e.g. CUDA backend
    availability for platform="CUDA"); this helper only does the
    registration.
    """
    if target_name in _registered_ffi_targets:
        return
    lib = _require_loaded_lib(registering_target=target_name)

    # Type registration. The C accessors return raw addresses we wrap into
    # capsules; XLA reads them as XLA_FFI_TypeId* / XLA_FFI_TypeInfo*.
    type_id_fn = getattr(lib, type_id_symbol)
    type_id_fn.restype = ctypes.c_void_p
    type_info_fn = getattr(lib, type_info_symbol)
    type_info_fn.restype = ctypes.c_void_p
    type_registration = {
        "type_id": _data_pycapsule(int(type_id_fn() or 0)),
        "type_info": _data_pycapsule(int(type_info_fn() or 0)),
    }
    type_key = f"{type_name}@{platform}"
    if type_key not in _registered_ffi_types:
        jax.ffi.register_ffi_type(type_name, type_registration, platform=platform)
        _registered_ffi_types.add(type_key)

    # Target registration: stage dict per the JAX FFI multi-stage API.
    inst_sym = getattr(lib, instantiate_symbol)
    exec_sym = getattr(lib, execute_symbol)
    fn_dict = {
        "instantiate": jax.ffi.pycapsule(inst_sym),  # pyright: ignore[reportUnknownMemberType,reportAny]
        "execute": jax.ffi.pycapsule(exec_sym),  # pyright: ignore[reportUnknownMemberType,reportAny]
    }
    jax.ffi.register_ffi_target(target_name, fn_dict, platform=platform)
    _registered_ffi_targets.add(target_name)


def is_available() -> bool:
    """True if the FFI shared library is loadable on this system."""
    _load_library()
    return _lib is not None


def ffi_capabilities() -> dict[str, bool]:
    """Return backend features compiled into the loaded FFI library."""
    _load_library()
    caps = 0
    if _lib is not None:
        cap_fn = _lib.pymdp_ffi_cpu_capabilities
        cap_fn.restype = ctypes.c_int32
        caps = int(cap_fn())  # pyright: ignore[reportAny]
    return {
        "aarch64_neon": bool(caps & _CAP_AARCH64_NEON),
        "arm_fp16_fml": bool(caps & _CAP_F16_FML),
        "cuda": bool(caps & _CAP_CUDA),
        "x86_avx2": bool(caps & _CAP_X86_AVX2),
        "x86_avx512f": bool(caps & _CAP_X86_AVX512F),
    }


def cpu_capabilities() -> dict[str, bool]:
    """Compatibility alias for ``ffi_capabilities()``."""
    return ffi_capabilities()


def load_error() -> Exception | None:
    """Return the load-time error, or None if the library is available."""
    _load_library()
    return _load_error


def flatten_ragged_f32(
    arrays: Sequence[jax.Array | np.ndarray],
) -> tuple[jnp.ndarray, np.ndarray]:
    """Cast arrays to f32 and concatenate them into one flat buffer.

    Returns (flat, offsets) where offsets is an int64 ndarray of length N+1
    giving the start/end of each element in the flat buffer.
    """
    offsets = np.zeros(len(arrays) + 1, dtype=np.int64)
    flat_chunks: list[jnp.ndarray] = []
    for i, arr in enumerate(arrays):
        a = jnp.asarray(arr, dtype=jnp.float32).reshape(-1)
        flat_chunks.append(a)
        offsets[i + 1] = offsets[i] + a.size
    flat = jnp.concatenate(flat_chunks) if flat_chunks else jnp.zeros(0, dtype=jnp.float32)
    return flat, offsets


def flatten_dep_lists(
    dep_lists: Sequence[Sequence[int]],
) -> tuple[np.ndarray, np.ndarray]:
    """Pack a ragged dependency list into (flat_int64, offsets_int64).

    Mirrors the FFI ABI: offsets is length N+1, flat concatenates each list in
    order. Both arrays are int64 to match the FFI attribute type.
    """
    offsets = np.zeros(len(dep_lists) + 1, dtype=np.int64)
    for i, deps in enumerate(dep_lists):
        offsets[i + 1] = offsets[i] + len(deps)
    flat = np.array([d for deps in dep_lists for d in deps], dtype=np.int64)
    return flat, offsets


def with_jax_grad(
    ffi_fn: Callable[..., T],
    jax_fn: Callable[..., T],
) -> Callable[..., T]:
    """Wrap an FFI forward call with a JAX-fallback backward via custom_vjp.

    `ffi_fn` runs in the forward pass; `jax_fn` is a numerically-equivalent
    JAX implementation used for the backward via ``jax.vjp``. Both must take
    the same positional arguments (the differentiable inputs); static config
    (policy_matrix, dependency lists, flags, ...) should be closed over.

    ``jax.ffi.ffi_call`` has no native JVP rule, so ``jax.grad`` through a
    raw FFI call fails. Routing the backward through ``jax_fn`` keeps
    gradients correct without sacrificing the FFI forward-pass speedup.
    """

    @jax.custom_vjp
    def fn(*inputs):  # pyright: ignore[reportMissingParameterType]
        return ffi_fn(*inputs)

    def fwd(*inputs):  # pyright: ignore[reportMissingParameterType]
        return ffi_fn(*inputs), inputs

    def bwd(residuals, g):  # pyright: ignore[reportMissingParameterType]
        _, vjp_fn = jax.vjp(jax_fn, *residuals)
        return vjp_fn(g)

    fn.defvjp(fwd, bwd)
    return fn


__all__ = [
    "MAX_FFI_DEP_RANK",
    "_register",
    "_register_multistage",
    "dep_indices_in_range",
    "dep_list_rank_ok",
    "cpu_capabilities",
    "ffi_capabilities",
    "flatten_dep_lists",
    "flatten_ragged_f32",
    "has_jax_cuda_backend",
    "is_available",
    "load_error",
    "with_jax_grad",
]
