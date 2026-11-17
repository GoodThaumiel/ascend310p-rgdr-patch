# SPDX-License-Identifier: Apache-2.0
"""Check or profile one grouped RGDR ablation on 310P3."""

import argparse
import os

import torch

from ascend310p_rgdr_patch import _LIBRARY_ENV

_PROBES = {"baseline": 0, "batch-pack": 1, "defer-write-wait": 2, "no-state-store": 3}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", *_PROBES), required=True)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--key-heads", type=int, choices=(8, 16), default=8)
    parser.add_argument("--cores", type=int, choices=range(1, 8), default=7)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true", help="run each launch through ACLGraph replay")
    parser.add_argument("--check", action="store_true", help="compare output and state instead of profiling")
    args = parser.parse_args()
    if not 1 <= args.batch <= 1024 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch must be in 1..1024, warmup non-negative and iterations positive")
    if args.check and args.variant in {"native", "baseline"}:
        parser.error("--check requires a probe variant")
    native = args.variant == "native"
    if native:
        from vllm_ascend.utils import bootstrap_custom_op_env

        bootstrap_custom_op_env()
    else:
        library = os.getenv(_LIBRARY_ENV)
        if not library:
            parser.error(f"set {_LIBRARY_ENV} to the rebuilt probe library")

    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    if native:
        from vllm_ascend.utils import enable_custom_op

        if not enable_custom_op():
            raise RuntimeError("The vLLM Ascend native op library could not be loaded")
    else:
        torch.ops.load_library(library)
        op = torch.ops.rgdr_v310_cube_fp16.run_.default
        if not any(argument.name == "probe" for argument in op._schema.arguments):
            raise RuntimeError("The installed RGDR library has no probes. Rebuild native-src.")

    batch, window, key_heads = args.batch, args.window, args.key_heads
    tokens, value_heads = batch * window, 2 * key_heads
    rng = torch.Generator().manual_seed(310023)

    def random(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).to(device=device, dtype=torch.float16)

    q = random((tokens, key_heads, 128), 0.1)
    k = random(q.shape, 0.1)
    v = random((tokens, value_heads, 128), 0.1)
    g = -random((tokens, value_heads), 0.03).abs()
    cpu_beta = (torch.randn((tokens, value_heads), generator=rng) * 0.2).sigmoid()
    beta = cpu_beta.to(device=device, dtype=torch.float16)
    initial = random((tokens + 1, value_heads, 128, 128), 0.02)
    state = initial.clone()
    lengths = torch.full((batch,), window, dtype=torch.int32, device=device)
    offsets = torch.arange(0, tokens + 1, window, dtype=torch.int32, device=device)
    slots = torch.arange(1, tokens + 1, dtype=torch.int32, device=device).view(batch, window)
    accepted = torch.arange(batch, dtype=torch.int32).remainder(window).add(1).to(device)
    out = torch.empty_like(v)
    scale = 128**-0.5

    if native:
        gate_fp32 = g.float()
        flat_slots = slots.view(-1)

    def launch(probe):
        if native:
            return torch.ops._C_ascend.npu_recurrent_gated_delta_rule_310(
                query=q,
                key=k,
                value=v,
                beta=beta,
                state=state,
                actual_seq_lengths=lengths,
                ssm_state_indices=flat_slots,
                g=gate_fp32,
                gk=None,
                num_accepted_tokens=accepted if window > 1 else None,
                scale_value=scale,
            )
        op(q, k, v, beta, g, state, slots, accepted, out, scale, False, args.cores, True, offsets, probe)
        return out

    def prepare(probe):
        if not args.graph:
            return lambda: launch(probe)
        state.copy_(initial)
        launch(probe)
        torch.npu.synchronize()
        state.copy_(initial)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = launch(probe)

        def replay():
            graph.replay()
            return graph_output

        return replay

    probe = _PROBES.get(args.variant, 0)
    if args.check:
        out.fill_(float("nan"))
        launch(0)
        torch.npu.synchronize()
        expected_out = out.cpu()
        if not expected_out.isfinite().all():
            raise RuntimeError("Grouped baseline did not produce finite output")
        expected_state = (initial if args.variant == "no-state-store" else state).cpu()
        run = prepare(probe)
        for _ in range(3):
            state.copy_(initial)
            out.fill_(float("nan"))
            run()
            torch.npu.synchronize()
            torch.testing.assert_close(out.cpu().view(torch.int16), expected_out.view(torch.int16))
            torch.testing.assert_close(state.cpu().view(torch.int16), expected_state.view(torch.int16))
        print(f"PASS {args.variant} B{batch} W{window} H{key_heads} cores={args.cores} repeats=3")
        return

    run = prepare(probe)
    for _ in range(args.warmup + args.iterations):
        state.copy_(initial)
        run()
    torch.npu.synchronize()
    print(f"Completed {args.warmup + args.iterations} launches of {args.variant} B{batch} W{window} H{key_heads}")


if __name__ == "__main__":
    main()
