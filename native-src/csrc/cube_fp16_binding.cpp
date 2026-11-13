// SPDX-License-Identifier: Apache-2.0
#include <acl/acl.h>
#include <torch/library.h>
#include <torch_npu/csrc/core/npu/NPUGuard.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>

#include "aclrtlaunch_rgdr_v310_cube_fp16.h"
#include "aclrtlaunch_rgdr_v310_cube_grouped_fp16.h"
#include "aclrtlaunch_rgdr_v310_cube_grouped_packed_fp16.h"

namespace {
void cube_fp16(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
               const at::Tensor& beta, const at::Tensor& gate, const at::Tensor& state,
               const at::Tensor& indices, const at::Tensor& accepted, const at::Tensor& out,
               double scale, bool deq, int64_t cores, bool grouped,
               const std::optional<at::Tensor>& row_offsets) {
  TORCH_CHECK(!grouped || !deq, "grouped RGDR uses one L0C drain and scales in UB");
  TORCH_CHECK(q.device().type() == c10::DeviceType::PrivateUse1, "NPU tensors required");
  for (const auto& tensor : {q, k, v, beta, gate, indices, accepted, out}) {
    TORCH_CHECK(tensor.device() == q.device() && tensor.is_contiguous(),
                "all tensors must be contiguous on the same NPU");
  }
  for (const auto& tensor : {q, k, v, beta, gate, state, out}) {
    TORCH_CHECK(tensor.scalar_type() == at::kHalf, "Cube RGDR data must be FP16");
  }
  const int64_t batch = accepted.numel();
  TORCH_CHECK(accepted.dim() == 1 && batch > 0 && batch <= 1024 &&
                  accepted.scalar_type() == at::kInt, "accepted must be int32[B]");
  TORCH_CHECK(q.dim() == 3 && q.size(1) == 8 && q.size(2) == 128 &&
                  q.sizes() == k.sizes(), "q/k must be [T,8,128]");
  const int64_t tokens = q.size(0);
  int64_t window;
  if (row_offsets) {
    TORCH_CHECK(grouped && row_offsets->device() == q.device() && row_offsets->is_contiguous() &&
                    row_offsets->scalar_type() == at::kInt && row_offsets->dim() == 1 &&
                    row_offsets->numel() == batch + 1, "packed RGDR requires int32 row_offsets[B+1]");
    TORCH_CHECK(indices.dim() == 2 && indices.size(0) == batch, "packed indices must be int32[B,W]");
    window = indices.size(1);
    TORCH_CHECK(tokens > 0 && tokens <= 1024 * 8, "invalid packed token count");
  } else {
    TORCH_CHECK(tokens % batch == 0, "uniform token count must be divisible by batch");
    window = tokens / batch;
    TORCH_CHECK(grouped || window == 4 || window == 8, "legacy Cube variants require MTP3/7");
  }
  TORCH_CHECK(window >= 1 && window <= 8, "Cube window must be in [1,8]");
  TORCH_CHECK(v.sizes() == at::IntArrayRef({tokens, 16, 128}) &&
                  v.sizes() == out.sizes(), "v/out must be [T,16,128]");
  TORCH_CHECK(beta.sizes() == at::IntArrayRef({tokens, 16}) && gate.sizes() == beta.sizes(),
              "beta/gate must be [T,16]");
  TORCH_CHECK(indices.scalar_type() == at::kInt &&
                  (row_offsets || (indices.dim() == 1 && indices.numel() == tokens)),
              "indices must be int32[T] for uniform or int32[B,W] for packed");
  TORCH_CHECK(state.dim() == 4 && state.size(0) > 1 &&
                  state.size(0) <= std::numeric_limits<int32_t>::max() && state.size(1) == 16 &&
                  state.size(2) == 128 && state.size(3) == 128 &&
                  state.device() == q.device(), "invalid state geometry or device");
  TORCH_CHECK(state.stride(3) == 1 && state.stride(2) == 128 && state.stride(1) == 128 * 128 &&
                  state.stride(0) >= 16 * 128 * 128 && state.stride(0) % 16 == 0 &&
                  state.storage_offset() % 16 == 0, "state pages must be 32-byte aligned with contiguous inner axes");
  TORCH_CHECK(row_offsets || state.is_contiguous(), "uniform Cube RGDR requires contiguous state");
  TORCH_CHECK(cores >= 1 && cores <= 7 && std::isfinite(scale) && scale > 0 && scale <= 1,
              "cores must be in [1,7] and scale in (0,1]");
  const c10_npu::OptionalNPUGuard guard(q.device());
  const auto stream = c10_npu::getCurrentNPUStream().stream();
#define RGDR_ARGS \
  static_cast<uint32_t>(cores), stream, q.data_ptr(), k.data_ptr(), v.data_ptr(), \
      beta.data_ptr(), gate.data_ptr(), state.data_ptr(), indices.data_ptr(), \
      accepted.data_ptr(), out.data_ptr(), static_cast<uint32_t>(batch), \
      static_cast<uint32_t>(window), static_cast<uint32_t>(state.size(0)), \
      static_cast<float>(scale)
  aclError result;
  if (row_offsets) {
    result = aclrtlaunch_rgdr_v310_cube_grouped_packed_fp16(
        RGDR_ARGS, row_offsets->data_ptr(), static_cast<uint32_t>(tokens),
        static_cast<uint64_t>(state.stride(0)));
  } else if (grouped) {
    result = aclrtlaunch_rgdr_v310_cube_grouped_fp16(RGDR_ARGS);
  } else {
    result = aclrtlaunch_rgdr_v310_cube_fp16(RGDR_ARGS, static_cast<uint32_t>(deq));
  }
#undef RGDR_ARGS
  TORCH_CHECK(result == ACL_SUCCESS, "FP16 Cube RGDR launch failed: ", result);
}
}  // namespace

TORCH_LIBRARY(rgdr_v310_cube_fp16, m) {
  m.def("run_(Tensor q, Tensor k, Tensor v, Tensor beta, Tensor gate, Tensor(a!) state, "
        "Tensor indices, Tensor accepted, Tensor(b!) out, float scale, bool deq, int cores=7, "
        "bool grouped=False, Tensor? row_offsets=None) -> ()");
}
TORCH_LIBRARY_IMPL(rgdr_v310_cube_fp16, PrivateUse1, m) {
  m.impl("run_", &cube_fp16);
}
TORCH_LIBRARY_IMPL(rgdr_v310_cube_fp16, Meta, m) {
  m.impl("run_", [](const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     const at::Tensor&, const at::Tensor&, const at::Tensor&,
                     double, bool, int64_t, bool, const std::optional<at::Tensor>&) {});
}
