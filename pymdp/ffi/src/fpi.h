// ABI declarations consumed by xla_register.cc — keep byte-identical with
// FpiCpu in fpi.cc and pymdp/ffi/_fpi.py buffer / attribute packing.

#pragma once

#include <cstdint>

#include "error_helpers.h"  // FfiError / FfiF32Buf / FfiF32Out / FfiInt64Span

namespace pymdp_ffi {

FfiError FpiCpu(FfiF32Buf ll_flat, FfiF32Buf lp_flat, FfiF32Out q_out, FfiInt64Span S, FfiInt64Span ll_offsets,
                FfiInt64Span lp_offsets, FfiInt64Span A_dep_flat, FfiInt64Span A_dep_offsets, int32_t num_iter);

}  // namespace pymdp_ffi
