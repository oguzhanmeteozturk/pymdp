"""JAX FFI custom calls for pymdp hot paths.

Build the extension with:

    cmake -S pymdp/ffi -B pymdp/ffi/build -G "Unix Makefiles" \\
        -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3.14
    cmake --build pymdp/ffi/build

The resulting ``libpymdp_ffi.dylib`` / ``.so`` is expected at
``pymdp/ffi/build/``. Set ``PYMDP_FFI_LIB`` to override the path.

This module is safe to import even if the shared library is missing —
``is_available()`` reports status, and callers should dispatch to a JAX
fallback when it is False.
"""

from __future__ import annotations

from ._core import cpu_capabilities, ffi_capabilities, is_available, load_error, with_jax_grad
from ._efe import (
    FLAG_USE_INDUCTIVE,
    FLAG_USE_PARAM_INFO_GAIN,
    FLAG_USE_STATES_INFO_GAIN,
    FLAG_USE_UTILITY,
    can_handle,
    neg_efe_all_policies_ffi,
)
from ._fpi import can_handle_fpi, fpi_ffi


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
