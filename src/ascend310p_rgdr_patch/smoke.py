# SPDX-License-Identifier: Apache-2.0
"""Compare the W=1 dispatch with native on the target 310P3 image."""

import argparse
from unittest.mock import Mock

import torch
import torch_npu

from ascend310p_rgdr_patch import _wrap, register


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    from vllm_ascend._310p.ops.fla import gdn_310
    from vllm_ascend.utils import enable_custom_op

    if not enable_custom_op():
        raise RuntimeError("Native Ascend extension is unavailable")
    register()
    grouped = gdn_310.npu_recurrent_gated_delta_rule_310
    if not getattr(grouped, "_ascend310p_grouped_rgdr", False):
        raise RuntimeError("Set ASCEND310P_RGDR_PATCH_LIBRARY before running the smoke test")

    native = grouped.__wrapped__
    launch = Mock(wraps=torch.ops.rgdr_v310_cube_fp16.run_.default)
    grouped = _wrap(native, gdn_310.l2norm_310p, launch, "decode", torch)
    rng = torch.Generator().manual_seed(310023)

    def random(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).to(device=device, dtype=torch.float16)

    for batch in (1, 32):
        q = random((1, batch, 8, 128), 0.1)
        k = random(q.shape, 0.1)
        v = random((1, batch, 16, 128), 0.1)
        g = -random((1, batch, 16), 0.03).abs()
        beta = random(g.shape, 0.2).sigmoid()
        offsets = torch.arange(batch + 1, device=device, dtype=torch.int32)
        slots = torch.arange(batch, 0, -1, device=device, dtype=torch.int32)
        initial = random((batch + 1, 16, 128, 128), 0.02)
        native_state = initial.clone()
        grouped_state = initial.clone()
        tail = {
            "cu_seqlens": offsets,
            "ssm_state_indices": slots,
            "use_qk_l2norm_in_kernel": False,
        }

        launch.reset_mock()
        output = grouped(q, k, v, g, beta, grouped_state, **tail)
        launch.assert_called_once()
        reference = native(q, k, v, g, beta, native_state, **tail)
        torch.npu.synchronize()

        out_cpu, ref_cpu = output.cpu(), reference.cpu()
        state_cpu, native_cpu = grouped_state.cpu(), native_state.cpu()
        torch.testing.assert_close(out_cpu, ref_cpu, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(state_cpu, native_cpu, atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(state_cpu[0], initial[0].cpu(), atol=0, rtol=0)
        print(
            f"B{batch} W1 PASS "
            f"out_max_abs={(out_cpu.float() - ref_cpu.float()).abs().max().item():.6g} "
            f"state_max_abs={(state_cpu.float() - native_cpu.float()).abs().max().item():.6g}"
        )


if __name__ == "__main__":
    main()
