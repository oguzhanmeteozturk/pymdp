cmake_minimum_required(VERSION 3.18)
project(pymdp_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Emit compile_commands.json so clang-tidy (driven from the repo-root Makefile)
# can resolve includes and flags without re-deriving the build invocation.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Without this, CMake picks no optimization flag when CMAKE_BUILD_TYPE is unset
# and our hand-rolled NEON helpers run unvectorized -- 10x slower than cblas.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# LTO/IPO. On AppleClang this picks -flto=thin; on GCC it picks -flto with the
# linker plugin. Default ON since both production targets (M3 Max, Jetson Nano)
# benefit from cross-TU inlining of the shared NEON helpers in
# kernel_primitives.h that are #included into both neg_efe_cpu_*.cc and fpi_cpu_runner.cc.
# Disable with -DPYMDP_FFI_LTO=OFF for faster iteration builds.
option(PYMDP_FFI_LTO "Enable link-time optimization (LTO)" ON)

# CUDA kernels — Jetson Nano (Maxwell SM 5.3) and beyond. Off by default on
# every host since nvcc / CUDA toolkit aren't present on most dev machines.
# Selected at build time via -DPYMDP_FFI_CUDA=ON; at runtime via
# PYMDP_FFI_USE_CUDA=1. When disabled the symbol
# isn't registered and the env var has no effect.
option(PYMDP_FFI_CUDA "Build CUDA neg-EFE kernel (Jetson Nano sm_53 baseline)" OFF)

find_package(Python COMPONENTS Interpreter REQUIRED)

# Locate the XLA FFI headers that ship with jaxlib.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _jax_ffi_rc
)
if(NOT _jax_ffi_rc EQUAL 0)
    message(FATAL_ERROR "Failed to query jax.ffi.include_dir()")
endif()
message(STATUS "JAX FFI include dir: ${JAX_FFI_INCLUDE_DIR}")

# Plain shared library; loaded at runtime with ctypes. No Python linkage needed —
# the C symbol is wrapped into a PyCapsule by jax.ffi.pycapsule.
set(_pymdp_ffi_sources
    src/neg_efe/neg_efe_cpu_entry.cc
    src/neg_efe/neg_efe_cpu_pipeline.cc
    src/registration/xla_register.cc
    src/fpi/fpi_cpu_runner.cc
)
# enable_language(CUDA) must run before add_library so .cu sources are
# recognized. Scoped to the PYMDP_FFI_CUDA branch so non-Jetson builds don't
# require a CUDA toolchain.
if(PYMDP_FFI_CUDA)
    # Host-side glue (FFI entry point, runtime API calls like
    # cudaGetDeviceProperties) lives in neg_efe_cuda_{cache,launch,runtime,entry}.cc
    # — compiled by the host C++ compiler so they can include XLA FFI api.h.
    # Device kernels live in neg_efe_cuda_kernels.cu and only see a small
    # launch-wrapper header (neg_efe_cuda_kernels.h). Splitting the TUs is required:
    # XLA FFI's api.h is C++17-heavy and nvcc 10.2's frontend rejects it
    # ("qualified name is not allowed" on parametrized template member
    # access).
    # Default to host-GPU arch detection (CMake 3.24+ understands `native` and
    # asks nvcc to probe the local devices). Falls back to sm_53 (Jetson Nano
    # baseline) on older CMake. Override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
    #
    # MUST run before enable_language(CUDA) — enable_language calls into
    # CMakeCUDAInformation, which sets CMAKE_CUDA_ARCHITECTURES to its own
    # default (sm_52 with nvcc 11/12). After that point the `NOT DEFINED`
    # check below would always be false, and `native` would never fire —
    # leaving Orin (sm_87) / ws0 (sm_86) builds JIT'ing from a Maxwell
    # baseline at runtime. Hoisting the resolution above enable_language
    # lets us bypass the CMake default and request `native` (or the user's
    # -DCMAKE_CUDA_ARCHITECTURES override) cleanly.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 53)
        endif()
    endif()
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    list(APPEND _pymdp_ffi_sources
        src/neg_efe/neg_efe_cuda_cache.cc
        src/neg_efe/neg_efe_cuda_entry.cc
        src/neg_efe/neg_efe_cuda_launch.cc
        src/neg_efe/neg_efe_cuda_runtime.cc
        src/neg_efe/neg_efe_cuda_kernels.cu
        src/neg_efe/neg_efe_cuda_context.h
        src/neg_efe/neg_efe_cuda_internal.h
        src/fpi/fpi_cuda_cache.cc
        src/fpi/fpi_cuda_host_shim.cc
        src/fpi/fpi_cuda_kernels.cu
        src/fpi/fpi_cuda_kernels.h
        src/fpi/fpi_cuda_runtime.cc
    )
endif()
add_library(pymdp_ffi SHARED ${_pymdp_ffi_sources})
target_include_directories(pymdp_ffi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
# Belt-and-suspenders: even for Release, force -O3 + target-native codegen so
# the NEON intrinsics in kernel_primitives.h are fully inlined. Scoped to this
# target (not add_compile_options) so -ffast-math doesn't leak into a parent
# CMake build that includes us — fast-math changes IEEE semantics and that
# surprise is unwelcome elsewhere.
# Host C++ only — nvcc takes a different flag spelling (-Xcompiler / --use_fast_math)
# and would error on bare -ffast-math when compiling .cu sources. CUDA-specific
# flags are added below in the PYMDP_FFI_CUDA block.
target_compile_options(pymdp_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math>
)

# Per-target codegen. On Apple silicon, AppleClang's default already targets the
# host (apple-m1+), so we leave -mcpu unset. On non-Apple aarch64 we default to
# `-mcpu=native` so the compiler picks up the host part — Cortex-A57 (Jetson
# Nano, v8.0) gets in-order scheduling, Cortex-A78AE (Orin NX, v8.2) gets the
# 4-wide out-of-order tune plus DOTPROD and FP16 vector arithmetic. Override
# with -DPYMDP_FFI_AARCH64_MCPU=cortex-a57 for portable Jetson-Nano-class
# builds (or any cross-compile where the build host isn't the run host).
# Pinning to cortex-a57 across all aarch64 — the prior default — was a
# substantial regression on A78AE: it locked the scheduler to A57's in-order
# pipeline and dropped v8.2 ISA features the host actually supports.
if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PYMDP_FFI_AARCH64_MCPU "native" CACHE STRING
        "GCC -mcpu target for aarch64 builds (default 'native'; set to e.g. cortex-a57 for portable Jetson Nano builds).")
    message(STATUS "Targeting aarch64 -mcpu=${PYMDP_FFI_AARCH64_MCPU}")
    # Host C++ only — same reasoning as the -O3/-ffast-math scope above.
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mcpu=${PYMDP_FFI_AARCH64_MCPU}>
    )
    target_link_options(pymdp_ffi PRIVATE -mcpu=${PYMDP_FFI_AARCH64_MCPU})
endif()

# x86_64 hosts: enable host-native codegen so AVX2/FMA/AVX-512 intrinsics in
# kernel_primitives.h compile and the compiler unlocks the matching auto-vec
# paths for the scalar fallbacks. Default ON; turn off for portable builds.
# -march=native sets both ISA-availability and tuning to the build host; the
# resulting binary is not portable to older x86 microarchitectures.
option(PYMDP_FFI_X86_NATIVE "Build with -march=native on x86_64 hosts" ON)
# AVX-512F intrinsics in kernel_primitives.h are opt-in even when the build
# host defines __AVX512F__: the AVX-512 path lost to AVX2 on every pymdp
# fixture on Ice Lake-SP (frequency throttle on sustained 512-bit FMA + slow
# horizontal-reduce on small inner-n sgemv). Opt in for Sapphire Rapids /
# Zen 4 hosts where the throttle is lighter; benchmark before flipping.
option(PYMDP_FFI_ENABLE_AVX512 "Compile in AVX-512F SIMD path on x86_64 hosts" OFF)
if(PYMDP_FFI_X86_NATIVE AND CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    message(STATUS "Targeting host x86 microarch (-march=native)")
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-march=native>
    )
    target_link_options(pymdp_ffi PRIVATE -march=native)
    if(PYMDP_FFI_ENABLE_AVX512)
        message(STATUS "AVX-512F SIMD path: enabled (opt-in)")
        target_compile_definitions(pymdp_ffi PRIVATE PYMDP_FFI_ENABLE_AVX512=1)
    endif()
endif()

if(PYMDP_FFI_LTO)
    include(CheckIPOSupported)
    # When CUDA is enabled CMake's default IPO check picks the first
    # language and fails on nvcc 10.2 (CMake doesn't know how to do IPO
    # for that toolchain). Pass LANGUAGES CXX so the host compiler is the
    # one being checked — that's the only language that benefits from
    # cross-TU inlining of kernel_primitives.h. CUDA opts out separately.
    if(PYMDP_FFI_CUDA)
        check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err LANGUAGES CXX)
    else()
        check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
    endif()
    if(_ipo_ok)
        message(STATUS "LTO: enabled")
        set_property(TARGET pymdp_ffi PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if(PYMDP_FFI_CUDA)
            # IPO TRUE on the target applies to all languages; opt CUDA
            # out explicitly so nvcc 10.2 doesn't try.
            set_property(TARGET pymdp_ffi PROPERTY INTERPROCEDURAL_OPTIMIZATION_CUDA FALSE)
        endif()
    else()
        message(WARNING "LTO requested but not supported by this toolchain: ${_ipo_err}")
    endif()
endif()

# SYSTEM include treats JAX FFI headers as third-party so their -Wall/-Wextra
# noise (unused params, sign compares, GNU __VA_ARGS__ pasting) doesn't surface
# when we tighten warnings on our own code.
target_include_directories(pymdp_ffi SYSTEM PRIVATE "${JAX_FFI_INCLUDE_DIR}")

# OpenMP — AppleClang requires the keg-only libomp from homebrew.
if(APPLE)
    execute_process(
        COMMAND brew --prefix libomp
        OUTPUT_VARIABLE LIBOMP_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _libomp_rc
    )
    if(NOT _libomp_rc EQUAL 0 OR NOT EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(FATAL_ERROR
            "libomp not found. Install with: brew install libomp")
    endif()
    message(STATUS "libomp prefix: ${LIBOMP_PREFIX}")
    target_compile_options(pymdp_ffi PRIVATE
        -Xclang -fopenmp
        -I${LIBOMP_PREFIX}/include
    )
    target_link_libraries(pymdp_ffi PRIVATE
        "${LIBOMP_PREFIX}/lib/libomp.dylib"
    )
    set_target_properties(pymdp_ffi PROPERTIES
        SUFFIX ".dylib"
        LINK_FLAGS "-Wl,-undefined,dynamic_lookup -Wl,-rpath,${LIBOMP_PREFIX}/lib"
    )
else()
    find_package(OpenMP REQUIRED)
    target_link_libraries(pymdp_ffi PRIVATE OpenMP::OpenMP_CXX)
endif()

if(PYMDP_FFI_CUDA)
    # CUDAToolkit (CMake 3.17+) is the modern way to find cudart. CMake
    # also auto-links cudart when CUDA language is enabled, so the explicit
    # link is belt-and-suspenders for the host .cc which calls runtime API
    # directly. Add CUDA::cudart_static if a self-contained dylib becomes
    # important; for now the shared cudart on the JetPack loader path is
    # the right call.
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pymdp_ffi PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_definitions(pymdp_ffi PRIVATE PYMDP_FFI_HAS_CUDA=1)
    # Match the host -ffast-math choice on device code. CMake already
    # passes -O3 via CMAKE_CUDA_FLAGS_RELEASE — adding our own here would
    # trigger `nvcc fatal: redefinition of argument 'optimize'`.
    # --use_fast_math drives PTX-level fast-math transforms (fused FMAs,
    # rcp/rsqrt approximations).
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    # ptxas -v: per-kernel register usage, spill stores/loads, smem,
    # constant-mem, and SASS bytes. Goes to stderr at compile time. Default
    # ON because the pymdp CUDA kernels are register-pressure-sensitive
    # (see feedback_fpi_cuda_no_op_changes: a "logically free" loop
    # reorganization regressed fpi_large by 9% via codegen perturbation
    # that ptxas output would have flagged before bench). One line per
    # __global__ kernel — cheap to read, expensive not to have.
    option(PYMDP_FFI_CUDA_PTXAS_VERBOSE
        "Pass -Xptxas=-v so nvcc reports per-kernel regs/spills/smem/SASS bytes" ON)
    if(PYMDP_FFI_CUDA_PTXAS_VERBOSE)
        target_compile_options(pymdp_ffi PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        )
        message(STATUS "ptxas verbose: enabled (per-kernel regs/spills/smem on every CUDA build)")
    endif()
    message(STATUS "CUDA kernels: enabled (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
endif()
