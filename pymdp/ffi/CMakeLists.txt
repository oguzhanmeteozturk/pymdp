cmake_minimum_required(VERSION 3.18)
project(pymdp_ffi LANGUAGES CXX)

# Build the shared pymdp FFI library used by JAX/XLA at runtime.
# This library is loaded through ctypes and wrapped in a PyCapsule on the
# Python side, so it does not need direct Python linkage.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Export compile_commands.json for tools like clang-tidy so they can inspect
# the same include paths and flags used by this build.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default to Release when no build type is provided. This ensures the NEON
# kernels are optimized instead of falling back to an unvectorized code path.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Enable link-time optimization by default. LTO lets the shared NEON helper
# code in kernel_primitives.h be cross-module inlined across the FFI target.
# On AppleClang this becomes thin LTO; on GCC it enables the linker plugin.
# Disable with -DPYMDP_FFI_LTO=OFF for faster edit/compile cycles.
option(PYMDP_FFI_LTO "Enable link-time optimization (LTO)" ON)

# Enable optional CUDA kernel support for Jetson Nano and newer devices.
# This is off by default because most development hosts do not have nvcc / the
# CUDA toolkit installed. Use -DPYMDP_FFI_CUDA=ON to build, and set
# PYMDP_FFI_USE_CUDA=1 at runtime to activate the CUDA path.
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

# Build a shared library loaded via ctypes. Python integration happens through
# jax.ffi.pycapsule, so the library does not need direct Python linkage.
set(_pymdp_ffi_sources
    src/neg_efe_cpu.cc
    src/xla_register.cc
    src/fpi.cc
)
# Enable CUDA only when requested. CUDA source files must be known before
# add_library, so enable_language(CUDA) is called inside the branch.
# This keeps non-CUDA builds independent of the CUDA toolchain.
if(PYMDP_FFI_CUDA)
    # Split CUDA into host glue and device kernels. The host-side file
    # neg_efe_cuda.cc is built with the host C++ compiler so it can include
    # XLA FFI api.h, while the device kernels in neg_efe_cuda_kernels.cu only
    # depend on a small launch-wrapper header.
    # This avoids nvcc frontend failures when compiling XLA FFI headers.
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    # Default to host-GPU arch detection (CMake 3.24+ understands `native` and
    # asks nvcc to probe the local devices). Falls back to sm_53 (Jetson Nano
    # baseline) on older CMake. Override with -DCMAKE_CUDA_ARCHITECTURES=86 etc.
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set(CMAKE_CUDA_ARCHITECTURES native)
        else()
            set(CMAKE_CUDA_ARCHITECTURES 53)
        endif()
    endif()
    list(APPEND _pymdp_ffi_sources
        src/neg_efe_cuda.cc
        src/neg_efe_cuda_kernels.cu
        src/neg_efe_cuda_memory.h
        src/neg_efe_cuda_context.h
    )
endif()
add_library(pymdp_ffi SHARED ${_pymdp_ffi_sources})
# Force -O3 and fast-math for host C++ compilation only, scoped to this
# target so the flags do not propagate to parent projects. CUDA uses a
# different device-specific fast-math flag below.
target_compile_options(pymdp_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math>
)

# Per-target codegen. On Apple silicon, AppleClang's default already targets the
# host (apple-m1+), so we leave -mcpu unset and let the compiler pick. On
# non-Apple aarch64 we assume Jetson Nano (Cortex-A57, ARMv8.0 — only baseline
# NEON, no fp16 vector arithmetic). -mcpu=cortex-a57 enables tuning for the
# in-order pipeline and matches PYMDP_FFI_DISABLE_F16=1 expectations in
# kernel_primitives.h. Pass to the linker too so LTO codegen sees it.
if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    message(STATUS "Targeting Cortex-A57 (Jetson Nano)")
    # Host C++ only — same reasoning as the -O3/-ffast-math scope above.
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mcpu=cortex-a57>
    )
    target_link_options(pymdp_ffi PRIVATE -mcpu=cortex-a57)
endif()

# Enable host-native x86_64 codegen by default so the compiler can use the
# local microarchitecture's vector instruction set. Turn this off for portable
# or cross-build scenarios.
option(PYMDP_FFI_X86_NATIVE "Build with -march=native on x86_64 hosts" ON)
# AVX-512F remains opt-in because it can be slower than AVX2 on many systems.
# Enable only on hosts with a well-behaved AVX-512 implementation after
# benchmarking.
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
    # Find and link the CUDA runtime and cuBLAS libraries. The explicit link
    # is a safety net for the host-side CUDA glue code that calls runtime APIs.
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
    message(STATUS "CUDA kernels: enabled (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
endif()
