// Shared entry-point glue for the three NegEfe FFI handlers (NegEfeCpu,
// NegEfeCudaHost, NegEfeCudaDev). Bundles the layout parse + runtime validate
// + param-info-gain buffer check + pA/pB present-flag extraction every entry
// point performs before dispatching its specific pipeline.

#pragma once

#include <cstdint>

#include "xla/ffi/api/ffi.h"

#include "common/error_helpers.h"
#include "neg_efe/neg_efe_layout.h"

namespace pymdp_ffi {

struct ParsedCall {
  CallShape     shape;
  Layout        L;
  KernelFlags   flags;
  BufferStrides strides;
  EpsilonLayout epsilon;
  bool          pA_present;
  bool          pB_present;
};

inline FfiError parse_and_validate_call(FfiS32Buf policy_matrix, FfiF32Buf qs_init, FfiF32Buf A, FfiF32Buf B,
                                        FfiF32Buf C, FfiF32Buf I, FfiF32Buf pA, FfiF32Buf pB,
                                        FfiF32Buf inductive_epsilon, FfiF32Out out, const LayoutSpans& spans,
                                        int32_t flags, ParsedCall* pc) {
  PYMDP_TRY(parse_neg_efe_layout(policy_matrix, spans, &pc->shape, &pc->L));
  pc->flags = KernelFlags::from_bits(flags);
  PYMDP_TRY(validate_neg_efe_runtime(pc->shape, pc->L, policy_matrix, qs_init, A, B, C, I, inductive_epsilon, out,
                                     &pc->strides, &pc->epsilon));
  PYMDP_TRY(validate_neg_efe_param_info_gain_buffers(pc->flags.use_param_info_gain, pc->shape.Bn, pc->strides,
                                                     pA.element_count(), pB.element_count()));
  pc->pA_present = pA.element_count() > 0;
  pc->pB_present = pB.element_count() > 0;
  return FfiError::Success();
}

}  // namespace pymdp_ffi
