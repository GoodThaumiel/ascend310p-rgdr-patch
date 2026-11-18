// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "aclrtlaunch_rgdr_v310_l0c_broadcast_probe.h"

namespace {
void l0c_broadcast(const at::Tensor& input, const at::Tensor& output) {
  TORCH_CHECK(input.device().type() == c10::DeviceType::PrivateUse1 &&
                  output.device() == input.device(),
              "probe requires tensors on the same NPU");
  TORCH_CHECK(input.is_contiguous() && input.scalar_type() == at::kHalf &&
                  input.sizes() == at::IntArrayRef({16}),
              "input must be contiguous FP16 [16]");
  TORCH_CHECK(output.is_contiguous() && output.scalar_type() == at::kHalf &&
                  output.sizes() == at::IntArrayRef({16, 16}),
              "output must be contiguous FP16 [16,16]");
  const c10_npu::OptionalNPUGuard guard(input.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
  const aclError result =
      aclrtlaunch_rgdr_v310_l0c_broadcast_probe(1, stream, input.data_ptr(), output.data_ptr());
  TORCH_CHECK(result == ACL_SUCCESS, "UB to L0C broadcast probe launch failed: ", result);
}
}  // namespace

TORCH_LIBRARY(rgdr_v310_l0c_probe, m) {
  m.def("run_(Tensor input, Tensor(a!) output) -> ()");
}

TORCH_LIBRARY_IMPL(rgdr_v310_l0c_probe, PrivateUse1, m) {
  m.impl("run_", &l0c_broadcast);
}
