"""JAX FFI custom calls for pymdp hot paths.

Build the extension with:

    uv sync --group build
    make ffi-build

Settings live in ``pyproject.toml`` under ``[tool.pymdp.ffi]``.
``CMAKE_FLAGS="-DPYMDP_FFI_CUDA=ON" make ffi-build`` for CUDA builds.

The resulting ``libpymdp_ffi.dylib`` / ``.so`` is expected at
``pymdp/ffi/build/``. Set ``PYMDP_FFI_LIB`` to override the path.

This module is safe to import even if the shared library is missing —
``is_available()`` reports status, and callers should dispatch to a JAX
fallback when it is False.
"""

from __future__ import annotations

import os
import sys

# Orin-only: turn XLA's GEMM/conv autotuner off (level 0). pymdp's HLO under
# PYMDP_FFI_USE_CUDA contains one custom-call (our FFI, not autotuned by XLA)
# plus a handful of elementwise / reduce fusions — no shapes the autotuner
# can pick beneficially between. Two effects on Ampere sm_87 with autotune
# at the default level 4:
#   1. The autotuner's delay-kernel timing bracket exceeds the actual fused-
#      reduce runtime (output is a 4-float vector from a (4, 1728) input),
#      producing noisy `cuda_timer.cc:87 Delay kernel timed out` warnings
#      and an unusable measurement that falls back to a default algorithm.
#   2. When autotune does pick, it favors cuBLAS/cuDNN dispatches with
#      higher per-call overhead than the lean kernels XLA emits at level 0
#      — measured +21% on rollout_loop (39.9 → 48.4 ms) on Orin.
# Setting level=0 skips the timing pass entirely. Opt out (e.g. for
# workloads that DO have autotuneable shapes in their HLO) via
# PYMDP_DISABLE_XLA_AUTOTUNE_OFF=1.
_AUTOTUNE_OFF = "--xla_gpu_autotune_level=0"


def _append_xla_flag(flag: str) -> None:
    """Append `flag` to XLA_FLAGS if not already present (whitespace-split
    membership). Safe under arbitrary user-supplied XLA_FLAGS."""
    existing = os.environ.get("XLA_FLAGS", "")
    if flag in existing.split():
        return
    # If the user is already setting the same option key (e.g. their own
    # --xla_gpu_autotune_level=N), don't override.
    key = flag.split("=", 1)[0]
    if key + "=" in existing:
        return
    os.environ["XLA_FLAGS"] = (existing + " " + flag).strip()


def _install_xla_autotune_off() -> None:
    if os.environ.get("PYMDP_FFI_USE_CUDA") != "1":
        return
    if os.environ.get("PYMDP_DISABLE_XLA_AUTOTUNE_OFF") == "1":
        return
    _append_xla_flag(_AUTOTUNE_OFF)


# Order matters: any XLA / JAX env var must be set before jax reads them,
# which happens on first backend init (typically the first device query
# or first JIT compile). The call below lands before `from ._core import ...`
# triggers any such initialization.
#
# Note on memory: on Orin (15 Gi shared LPDDR), JAX's default ~75% client
# preallocation eats host RAM just as hard as it would eat VRAM on a
# discrete card. Despite that, leaving JAX's default preallocation on is
# the right perf choice — pymdp's many small JAX dispatches per rollout
# step pay tens of µs per buffer create under on-demand allocation.
# Measured -18% on rollout_loop when we kept preallocation on vs forcing
# it off (39.9 ms vs 48.7 ms). Set XLA_PYTHON_CLIENT_PREALLOCATE=false
# manually if a workload's parallel memory pressure outweighs the
# per-dispatch alloc cost.
_install_xla_autotune_off()


from ._core import cpu_capabilities, ffi_capabilities, is_available, load_error, with_jax_grad
from ._efe import (
    FLAG_USE_INDUCTIVE,
    FLAG_USE_PARAM_INFO_GAIN,
    FLAG_USE_STATES_INFO_GAIN,
    FLAG_USE_UTILITY,
    _register_all_efe_targets_eager,
    can_handle,
    neg_efe_all_policies_ffi,
)
from ._fpi import _register_all_fpi_targets_eager, can_handle_fpi, fpi_ffi


# Eager FFI registration — root-cause fix for the Mosaic-GPU
# CustomCallResources destructor crash on Tegra (Orin / Jetson) under
# PYMDP_FFI_USE_CUDA=1. Crash chain (captured under gdb, May 2026):
#
#   PJRT_LoadedExecutable_Destroy
#   → PjRtStreamExecutorLoadedExecutable::~PjRtStreamExecutorLoadedExecutable
#   → GpuExecutable::~GpuExecutable
#   → CustomCallThunk::~CustomCallThunk
#   → shared_ptr<ExecutionState>::_M_dispose
#   → TypeRegistry::GetTypeInfo<mosaic::gpu::CustomCallResources>::__invoke
#   → mosaic::gpu::CustomCallResources::~CustomCallResources
#   → stream_executor::DeviceAddressHandle::~DeviceAddressHandle  ← SIGSEGV
#
# When `_register_*` fires mid-JIT-trace (the old lazy-registration pattern,
# inside `neg_efe_all_policies_ffi` / `fpi_ffi` on first call), the FFI
# TypeRegistry's internal storage rehashes. Mosaic-GPU has already cached a
# reference to the registry by then, so the cached reference goes stale.
# The next destructor that walks through that reference (during pjit cache
# eviction, function GC, or process teardown) calls through it into freed
# memory and SIGSEGV's at the line above. Doing all registrations up front,
# before any JAX backend init / JIT trace, lets the TypeRegistry settle
# before Mosaic-GPU forms any caches — subsequent destructor walks then
# resolve through stable pointers.
#
# The lazy-registration calls inside `_efe.py` / `_fpi.py` are kept as
# harmless fallbacks (idempotent via the `@platform`-keyed dedup set) —
# they fire on first call in environments where eager registration
# silently failed (e.g. JAX version mismatch).
def _eager_register(fn, name: str) -> None:
    try:
        fn()
    except Exception as exc:
        # Surface to stderr (gated). Silent failure here would defeat the
        # whole point — eager-only is the root-cause fix; falling through
        # to lazy registration reintroduces the rehash mid-trace hazard.
        if os.environ.get("PYMDP_FFI_QUIET_EAGER_REGISTER") != "1":
            sys.stderr.write(f"pymdp.ffi: eager register {name} failed: {type(exc).__name__}: {exc}\n")


_eager_register(_register_all_efe_targets_eager, "neg-efe")
_eager_register(_register_all_fpi_targets_eager, "fpi")


__all__ = [
    "FLAG_USE_UTILITY",
    "FLAG_USE_STATES_INFO_GAIN",
    "FLAG_USE_INDUCTIVE",
    "FLAG_USE_PARAM_INFO_GAIN",
    "can_handle",
    "can_handle_fpi",
    "cpu_capabilities",
    "ffi_capabilities",
    "fpi_ffi",
    "is_available",
    "load_error",
    "neg_efe_all_policies_ffi",
    "with_jax_grad",
]
