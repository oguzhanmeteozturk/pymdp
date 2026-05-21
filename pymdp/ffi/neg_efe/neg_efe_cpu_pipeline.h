// Per-batch-element orchestration for NegEfeCpu: factor-history build (memoized
// across broadcast-vmap batch elements), level walk via Kernel::run_level,
// final scatter to the policy-score output buffer.

#pragma once

#include <cstdint>

#include "common/error_helpers.h"
#include "neg_efe/neg_efe_cpu_core.h"

namespace pymdp_ffi {

// One vmap-batch element: build factor histories (or hit the per-thread pm
// cache), walk levels, scatter to out. Returns Success on the happy path; the
// only failure mode is policy-matrix out-of-bounds detected inside
// build_factor_history_host.
FfiError process_batch_element(const Layout& L, int32_t flags, const BatchInputs& in, CallScratch& sc);

}  // namespace pymdp_ffi
