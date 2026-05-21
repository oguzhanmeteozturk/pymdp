# FFI build — settings live in pyproject.toml [tool.pymdp.ffi].

SHELL := /bin/bash

FFI_BUILD := uv run --group build python scripts/ffi_build.py

# Assumes the standard uv venv layout. Override with
# PYTHON=/path/to/python make ffi-build. CUDA is auto-detected via nvcc;
# force off with PYMDP_FFI_CUDA=OFF make ffi-build.

.PHONY: ffi-compile-db ffi-build ffi-clean cudnn-shim

ffi-compile-db:
	@$(FFI_BUILD) configure --quiet

ffi-build:
	@$(FFI_BUILD) build

ffi-clean:
	@$(FFI_BUILD) clean

# cudnn-shim plants the cuDNN sbsa runtime into the venv at the path the
# jax_plugins.xla_cuda12 loader probes (`nvidia.cudnn.__path__/lib/`). The
# pip wheel `nvidia-cudnn-cu12` is x86_64-only, so on aarch64 (Jetson, Orin)
# nothing lands there by default and the loader falls through to the system
# cuDNN, which on JetPack 6.2 is 9.3.0 — too old for sm_87 jaxlib built
# against 9.8. Symlinking the unpacked sbsa tarball into that namespace
# tricks the loader into preloading the right version via ctypes; subsequent
# dlopens of libcudnn.so.9 reuse the cached SONAME. The master lib's
# RUNPATH=$$ORIGIN means one symlink covers all sub-libs.
#
# Idempotent: re-run after `uv sync --reinstall` or a fresh `.venv`. Override
# CUDNN_VERSION / CUDNN_CACHE_DIR for non-default placement.
CUDNN_VERSION ?= 9.8.0.87
CUDNN_CACHE_DIR ?= $(HOME)/cudnn
CUDNN_TARBALL := cudnn-linux-sbsa-$(CUDNN_VERSION)_cuda12-archive.tar.xz
CUDNN_UNPACK_DIR := $(CUDNN_CACHE_DIR)/cudnn-linux-sbsa-$(CUDNN_VERSION)_cuda12-archive
CUDNN_URL := https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-sbsa/$(CUDNN_TARBALL)

cudnn-shim:
	@test -d .venv || { echo "no .venv at $(CURDIR)/.venv — run 'uv sync --extra gpu' first" >&2; exit 1; }
	@SP=$$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])'); \
	    test -n "$$SP" || { echo "could not resolve venv site-packages" >&2; exit 1; }; \
	    if [ ! -d "$(CUDNN_UNPACK_DIR)" ]; then \
	        mkdir -p "$(CUDNN_CACHE_DIR)"; \
	        if [ ! -f "$(CUDNN_CACHE_DIR)/$(CUDNN_TARBALL)" ]; then \
	            echo "fetching $(CUDNN_TARBALL) (~880MB)..."; \
	            curl -fL --progress-bar -o "$(CUDNN_CACHE_DIR)/$(CUDNN_TARBALL)" "$(CUDNN_URL)"; \
	        fi; \
	        echo "extracting to $(CUDNN_UNPACK_DIR)..."; \
	        tar xJf "$(CUDNN_CACHE_DIR)/$(CUDNN_TARBALL)" -C "$(CUDNN_CACHE_DIR)"; \
	    fi; \
	    test -f "$(CUDNN_UNPACK_DIR)/lib/libcudnn.so.9" || { echo "unpacked tree missing libcudnn.so.9 at $(CUDNN_UNPACK_DIR)/lib/" >&2; exit 1; }; \
	    mkdir -p "$$SP/nvidia/cudnn"; \
	    : > "$$SP/nvidia/cudnn/__init__.py"; \
	    ln -sfn "$(CUDNN_UNPACK_DIR)/lib" "$$SP/nvidia/cudnn/lib"; \
	    echo "shim ready: $$SP/nvidia/cudnn/lib -> $(CUDNN_UNPACK_DIR)/lib"
	@.venv/bin/python -c "import jax; assert jax.default_backend() == 'gpu', jax.default_backend(); import jax.numpy as jnp; (jnp.ones((8,8)) @ jnp.ones((8,8))).block_until_ready(); print('cudnn-shim verified: jax gpu backend ok on', jax.devices())"
