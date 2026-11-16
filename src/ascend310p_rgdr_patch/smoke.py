# SPDX-License-Identifier: Apache-2.0
"""Check W=1 grouped RGDR against native Ascend or a CPU reference."""

import argparse
import os
from unittest.mock import Mock

import torch

from ascend310p_rgdr_patch import _KEY_HEAD_COUNTS, _wrap, register


def _cpu_reference(q, k, v, gate, beta, initial, slots):
    """Compute a single recurrent update with FP32 CPU arithmetic."""
    slots = slots.cpu().long()
    heads_per_key = v.shape[-2] // k.shape[-2]
    q = q.cpu().float().repeat_interleave(heads_per_key, dim=-2)
    k = k.cpu().float().repeat_interleave(heads_per_key, dim=-2)
    state = initial.cpu().clone()
    decayed = state.index_select(0, slots).float() * gate[0].cpu().float().exp()[..., None, None]
    prediction = (decayed @ k[0].unsqueeze(-1)).squeeze(-1)
    delta = (v[0].cpu().float() - prediction) * beta[0].cpu().float().unsqueeze(-1)
    updated = decayed + delta.unsqueeze(-1) * k[0].unsqueeze(-2)
    state.index_copy_(0, slots, updated.to(state.dtype))
    output = (updated @ q[0].unsqueeze(-1)).squeeze(-1) * (q.shape[-1] ** -0.5)
    return output.unsqueeze(0).to(v.dtype), state


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--reference", choices=("native", "cpu"), default="native")
    parser.add_argument("--graph", action="store_true", help="capture grouped RGDR and verify ACLGraph replay")
    parser.add_argument("--key-heads", type=int, choices=_KEY_HEAD_COUNTS, default=8)
    args = parser.parse_args()

    if args.reference == "native":
        from vllm_ascend.utils import bootstrap_custom_op_env

        # Configure OPP discovery before initializing the device in this process.
        bootstrap_custom_op_env()

    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)

    if args.reference == "native":
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("Native Ascend extension is unavailable. Use --reference cpu to test grouped RGDR.")

    from vllm_ascend._310p.ops.fla import gdn_310

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

    key_heads = args.key_heads
    value_heads = 2 * key_heads
    for batch in (1, 32):
        q = random((1, batch, key_heads, 128), 0.1)
        k = random(q.shape, 0.1)
        v = random((1, batch, value_heads, 128), 0.1)
        g = -random((1, batch, value_heads), 0.03).abs()
        beta = random(g.shape, 0.2).sigmoid()
        inputs = (q, k, v, g, beta)
        offsets = torch.arange(batch + 1, device=device, dtype=torch.int32)
        slots = torch.arange(batch, 0, -1, device=device, dtype=torch.int32)
        initial = random((batch + 1, value_heads, 128, 128), 0.02)
        grouped_state = initial.clone()
        tail = {
            "cu_seqlens": offsets,
            "ssm_state_indices": slots,
            "use_qk_l2norm_in_kernel": False,
        }

        launch.reset_mock()
        output = grouped(*inputs, grouped_state, **tail)
        launch.assert_called_once()
        torch.npu.synchronize()

        if args.graph:
            grouped_state.copy_(initial)
            graph = torch.npu.NPUGraph()
            launch.reset_mock()
            with torch.npu.graph(graph):
                output = grouped(*inputs, grouped_state, **tail)
            launch.assert_called_once()
            grouped_state.copy_(initial)
            graph.replay()
            torch.npu.synchronize()

        _check_result(args.reference, batch, batch, output, grouped_state, initial, native, inputs, tail)
        if args.graph and batch > 1:
            active = batch // 2
            # Match vLLM's graph padding: repeat the last offset and use slot 0.
            # Change the active slots too, while retaining both captured addresses.
            offsets[active + 1 :].fill_(active)
            slots[:active].copy_(torch.arange(1, active + 1, dtype=torch.int32, device=device))
            slots[active:].zero_()
            grouped_state.copy_(initial)
            output.fill_(float("nan"))
            graph.replay()
            torch.npu.synchronize()
            launch.assert_called_once()
            _check_result(args.reference, batch, active, output, grouped_state, initial, native, inputs, tail)
        if args.graph:
            print(f"B{batch} W1 ACLGraph capture/replay PASS", flush=True)


def _check_result(reference_mode, batch, active, output, grouped_state, initial, native, inputs, tail):
    valid_inputs = tuple(tensor[:, :active] for tensor in inputs)
    slots = tail["ssm_state_indices"][:active]
    if reference_mode == "native":
        reference_state = initial.clone()
        try:
            reference = native(
                *valid_inputs,
                reference_state,
                cu_seqlens=tail["cu_seqlens"][: active + 1],
                ssm_state_indices=slots,
                use_qk_l2norm_in_kernel=False,
            )
            torch.npu.synchronize()
        except RuntimeError as error:
            raise RuntimeError(
                f"Grouped RGDR completed for B{batch}, but the native V310 reference failed. "
                f"ASCEND_CUSTOM_OPP_PATH={os.getenv('ASCEND_CUSTOM_OPP_PATH', '<unset>')}. "
                "Check the native custom_transformer OPP package, or run --reference cpu."
            ) from error
    else:
        reference, reference_state = _cpu_reference(*valid_inputs, initial, slots)

    out_cpu, ref_cpu = output[:, :active].cpu(), reference.cpu()
    state_cpu, ref_state_cpu = grouped_state.cpu(), reference_state.cpu()
    torch.testing.assert_close(out_cpu, ref_cpu, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state_cpu, ref_state_cpu, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(state_cpu[0], initial[0].cpu(), atol=0, rtol=0)
    padding = output[:, active:].cpu()
    torch.testing.assert_close(padding, torch.zeros_like(padding), atol=0, rtol=0)
    print(
        f"B{batch} W1 PASS active={active} heads={valid_inputs[0].shape[2]}/{output.shape[2]} "
        f"reference={reference_mode} "
        f"out_max_abs={(out_cpu.float() - ref_cpu.float()).abs().max().item():.6g} "
        f"state_max_abs={(state_cpu.float() - ref_state_cpu.float()).abs().max().item():.6g}"
    )


if __name__ == "__main__":
    main()
