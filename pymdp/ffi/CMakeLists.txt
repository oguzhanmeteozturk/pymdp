cmake_minimum_required(VERSION 3.18)
project(pymdp_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Unset CMAKE_BUILD_TYPE leaves no -O flag; NEON helpers then run ~10x slower.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

option(PYMDP_FFI_LTO "Enable link-time optimization (LTO)" ON)

# AUTO: nvcc on PATH. Runtime opt-in: PYMDP_FFI_USE_CUDA=1.
# CI/headless: pass -DCMAKE_CUDA_ARCHITECTURES=53|86|87 (native needs a local GPU).
set(PYMDP_FFI_CUDA "AUTO" CACHE STRING "Build CUDA kernels: ON, OFF, or AUTO (detect nvcc)")
set_property(CACHE PYMDP_FFI_CUDA PROPERTY STRINGS AUTO ON OFF)

string(TOUPPER "${PYMDP_FFI_CUDA}" _pymdp_cuda_mode)
if(_pymdp_cuda_mode STREQUAL "AUTO")
    find_program(_pymdp_nvcc nvcc)
    if(_pymdp_nvcc)
        set(_pymdp_ffi_build_cuda ON)
        message(STATUS "PYMDP_FFI_CUDA=AUTO: enabling CUDA (found nvcc: ${_pymdp_nvcc})")
    else()
        set(_pymdp_ffi_build_cuda OFF)
        message(STATUS "PYMDP_FFI_CUDA=AUTO: CUDA disabled (nvcc not found)")
    endif()
elseif(_pymdp_cuda_mode STREQUAL "ON")
    find_program(_pymdp_nvcc nvcc REQUIRED)
    set(_pymdp_ffi_build_cuda ON)
    message(STATUS "PYMDP_FFI_CUDA=ON: building CUDA kernels (nvcc: ${_pymdp_nvcc})")
else()
    set(_pymdp_ffi_build_cuda OFF)
    message(STATUS "PYMDP_FFI_CUDA=OFF: CUDA kernels excluded")
endif()

find_package(Python COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import jax.ffi; print(jax.ffi.include_dir())"
    OUTPUT_VARIABLE JAX_FFI_INCLUDE_DIR
    ERROR_VARIABLE _jax_ffi_err
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _jax_ffi_rc
)
if(NOT _jax_ffi_rc EQUAL 0)
    message(FATAL_ERROR
        "Failed to query jax.ffi.include_dir() (Python: ${Python_EXECUTABLE}).\n"
        "${_jax_ffi_err}\n"
        "Ensure jax/jaxlib are installed in the chosen interpreter.")
endif()
message(STATUS "JAX FFI include dir: ${JAX_FFI_INCLUDE_DIR}")

set(_pymdp_ffi_sources
    neg_efe/neg_efe_cpu_entry.cc
    neg_efe/neg_efe_cpu_pipeline.cc
    registration/xla_register.cc
    fpi/fpi_cpu_runner.cc
)
if(_pymdp_ffi_build_cuda)
    # Host .cc files include XLA FFI (C++17); kernels live in .cu (CUDA 14).
    # Set arch before enable_language — otherwise CMake defaults to sm_52.
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
        neg_efe/neg_efe_cuda_cache.cc
        neg_efe/neg_efe_cuda_entry.cc
        neg_efe/neg_efe_cuda_launch.cc
        neg_efe/neg_efe_cuda_runtime.cc
        neg_efe/neg_efe_cuda_kernels.cu
        fpi/fpi_cuda_cache.cc
        fpi/fpi_cuda_host_shim.cc
        fpi/fpi_cuda_kernels.cu
        fpi/fpi_cuda_runtime.cc
    )
endif()

add_library(pymdp_ffi SHARED ${_pymdp_ffi_sources})
set_target_properties(pymdp_ffi PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
)
target_include_directories(pymdp_ffi PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# Host C++ only — nvcc rejects bare -ffast-math / -march flags on .cu sources.
target_compile_options(pymdp_ffi PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-O3 -ffast-math -Wall -Wextra>
)

if(NOT APPLE AND CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(PYMDP_FFI_AARCH64_MCPU "native" CACHE STRING
        "GCC -mcpu for aarch64 (use cortex-a57 for portable/cross builds)")
    message(STATUS "Targeting aarch64 -mcpu=${PYMDP_FFI_AARCH64_MCPU}")
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-mcpu=${PYMDP_FFI_AARCH64_MCPU}>
    )
    target_link_options(pymdp_ffi PRIVATE -mcpu=${PYMDP_FFI_AARCH64_MCPU})
endif()

option(PYMDP_FFI_X86_NATIVE "Build with -march=native on x86_64 hosts" ON)
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
    if(_pymdp_ffi_build_cuda)
        check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err LANGUAGES CXX)
    else()
        check_ipo_supported(RESULT _ipo_ok OUTPUT _ipo_err)
    endif()
    if(_ipo_ok)
        message(STATUS "LTO: enabled")
        set_property(TARGET pymdp_ffi PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
        if(_pymdp_ffi_build_cuda)
            set_property(TARGET pymdp_ffi PROPERTY INTERPROCEDURAL_OPTIMIZATION_CUDA FALSE)
        endif()
    else()
        message(WARNING "LTO requested but not supported by this toolchain: ${_ipo_err}")
    endif()
endif()

target_include_directories(pymdp_ffi SYSTEM PRIVATE "${JAX_FFI_INCLUDE_DIR}")

if(APPLE)
    set(LIBOMP_PREFIX "" CACHE PATH "libomp install prefix (default: brew --prefix libomp)")
    if(NOT LIBOMP_PREFIX)
        execute_process(
            COMMAND brew --prefix libomp
            OUTPUT_VARIABLE LIBOMP_PREFIX
            OUTPUT_STRIP_TRAILING_WHITESPACE
            RESULT_VARIABLE _libomp_rc
            ERROR_VARIABLE _libomp_brew_err
            ERROR_STRIP_TRAILING_WHITESPACE
        )
        if(NOT _libomp_rc EQUAL 0)
            message(FATAL_ERROR
                "libomp not found. Install with: brew install libomp\n"
                "Or pass -DLIBOMP_PREFIX=/path/to/libomp")
        endif()
    endif()
    if(NOT EXISTS "${LIBOMP_PREFIX}/lib/libomp.dylib")
        message(FATAL_ERROR
            "libomp not found at ${LIBOMP_PREFIX}/lib/libomp.dylib\n"
            "Install with: brew install libomp, or pass -DLIBOMP_PREFIX=...")
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

if(_pymdp_ffi_build_cuda)
    find_package(CUDAToolkit REQUIRED)
    target_link_libraries(pymdp_ffi PRIVATE CUDA::cudart CUDA::cublas)
    target_compile_definitions(pymdp_ffi PRIVATE PYMDP_FFI_HAS_CUDA=1)
    target_compile_options(pymdp_ffi PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>
    )
    option(PYMDP_FFI_CUDA_PTXAS_VERBOSE
        "Pass -Xptxas=-v (per-kernel regs/spills/smem)" ON)
    if(PYMDP_FFI_CUDA_PTXAS_VERBOSE)
        target_compile_options(pymdp_ffi PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
        )
        message(STATUS "ptxas verbose: enabled")
    endif()
    message(STATUS "CUDA kernels: enabled (CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES})")
endif()
