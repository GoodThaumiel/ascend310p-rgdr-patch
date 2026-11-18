// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>

#include "aclrtlaunch_rgdr_v310_l0c_broadcast_probe.h"
#include "aclrtlaunch_rgdr_v310_l0c_broadcast.h"
#include "aclrtlaunch_rgdr_v310_l0c_broadcast_raw.h"
#include "aclrtlaunch_rgdr_v310_l0c_gather.h"
#include "aclrtlaunch_rgdr_v310_l0c_interleaved.h"
#include "aclrtlaunch_rgdr_v310_l0c_mmad.h"
#include "aclrtlaunch_rgdr_v310_l0c_serial.h"

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

void l0c_bench(const at::Tensor& delta, const at::Tensor& offsets, const at::Tensor& lhs,
               const at::Tensor& rhs, const at::Tensor& expanded, const at::Tensor& matrix,
               int64_t mode, int64_t repeats) {
  TORCH_CHECK(delta.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& tensor : {delta, offsets, lhs, rhs, expanded, matrix}) {
    TORCH_CHECK(tensor.device() == delta.device() && tensor.is_contiguous() &&
                    tensor.storage_offset() % 16 == 0,
                "probe requires aligned contiguous tensors on one NPU");
  }
  for (const auto& tensor : {delta, lhs, rhs, expanded, matrix}) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, "probe data must be FP16");
  }
  TORCH_CHECK(delta.sizes() == at::IntArrayRef({128}) &&
                  offsets.sizes() == at::IntArrayRef({2048}) && offsets.scalar_type() == at::kInt &&
                  lhs.sizes() == at::IntArrayRef({128 * 144}) &&
                  rhs.sizes() == at::IntArrayRef({144 * 16}) &&
                  expanded.sizes() == at::IntArrayRef({128, 16}) && matrix.sizes() == expanded.sizes(),
              "invalid L0C benchmark geometry");
  TORCH_CHECK(repeats >= 1 && repeats <= 256, "repeats must be in [1,256]");
  const c10_npu::OptionalNPUGuard guard(delta.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
#define L0C_CASE(ID, NAME) \
  case ID: \
    result = aclrtlaunch_rgdr_v310_l0c_##NAME( \
        1, stream, delta.data_ptr(), offsets.data_ptr(), lhs.data_ptr(), rhs.data_ptr(), \
        expanded.data_ptr(), matrix.data_ptr(), static_cast<uint32_t>(repeats)); \
    break
  aclError result;
  switch (mode) {
    L0C_CASE(0, gather);
    L0C_CASE(1, broadcast_raw);
    L0C_CASE(2, broadcast);
    L0C_CASE(3, mmad);
    L0C_CASE(4, serial);
    L0C_CASE(5, interleaved);
    default:
      TORCH_CHECK(false, "invalid L0C benchmark variant");
  }
#undef L0C_CASE
  TORCH_CHECK(result == ACL_SUCCESS, "L0C benchmark launch failed: ", result);
}
}  // namespace

TORCH_LIBRARY(rgdr_v310_l0c_probe, m) {
  m.def("run_(Tensor input, Tensor(a!) output) -> ()");
  m.def("bench_(Tensor delta, Tensor offsets, Tensor lhs, Tensor rhs, Tensor(a!) expanded, "
        "Tensor(b!) matrix, int mode, int repeats) -> ()");
}

TORCH_LIBRARY_IMPL(rgdr_v310_l0c_probe, PrivateUse1, m) {
  m.impl("run_", &l0c_broadcast);
  m.impl("bench_", &l0c_bench);
}
