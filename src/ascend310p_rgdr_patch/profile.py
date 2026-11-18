# SPDX-License-Identifier: Apache-2.0
"""Check or profile one grouped RGDR ablation on 310P3."""

import argparse
import os

import torch

from ascend310p_rgdr_patch import _DEFAULT_CORES, _LIBRARY_ENV, _MAX_CORES

_DIAGNOSTIC_PROBES = {
    "no-gate-scale": 4,
    "no-rank1": 5,
    "no-rank1-broadcast": 6,
    "no-rank1-mul": 7,
    "no-rank1-add": 8,
}
_LEGACY_BASELINE = 0
_GATHER16_INTERLEAVE = 11
_FUSED_PROBES = {"rank1-gather16-fused": 12, "rank1-gather16-interleave-fused": 13}
_PROBES = {
    "baseline": _GATHER16_INTERLEAVE,
    "legacy-baseline": _LEGACY_BASELINE,
    "batch-pack": 1,
    "defer-write-wait": 2,
    "no-state-store": 3,
    **_DIAGNOSTIC_PROBES,
    "rank1-brcb16": 9,
    "rank1-gather16": 10,
    "rank1-gather16-interleave": _GATHER16_INTERLEAVE,
    **_FUSED_PROBES,
}


def _assert_bits(actual, expected, case):
    actual = actual.cpu()
    actual_bits = actual.view(torch.int16)
    expected_bits = expected.view(torch.int16)
    if torch.equal(actual_bits, expected_bits):
        return

    mismatches = (actual_bits != expected_bits).nonzero()
    first = tuple(mismatches[0].tolist())
    raise AssertionError(
        f"{case}: {len(mismatches)}/{actual.numel()} FP16 elements differ; first={first}; "
        f"actual={actual[first].item():.8g} (0x{actual_bits[first].item() & 0xFFFF:04x}), "
        f"expected={expected[first].item():.8g} (0x{expected_bits[first].item() & 0xFFFF:04x})"
    )


def _check_fused(actual, expected, case):
    actual = actual.cpu().flatten()
    expected = expected.flatten()
    max_abs = 0.0
    squared_error = 0.0
    changed = 0
    for got, ref in zip(actual.split(1 << 20), expected.split(1 << 20), strict=True):
        changed += torch.count_nonzero(got.view(torch.int16) != ref.view(torch.int16)).item()
        diff = got.float() - ref.float()
        max_abs = max(max_abs, diff.abs().max().item())
        squared_error += diff.square().sum(dtype=torch.float64).item()
    rmse = (squared_error / actual.numel()) ** 0.5
    print(f"{case} changed={changed}/{actual.numel()} max_abs={max_abs:.6g} rmse={rmse:.6g}", flush=True)
    torch.testing.assert_close(actual, expected, atol=2**-11, rtol=2**-8, msg=lambda msg: f"{case}: {msg}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=("native", *_PROBES), default="baseline")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--window", type=int, choices=range(1, 9), default=4)
    parser.add_argument("--key-heads", type=int, choices=(8, 16), default=8)
    parser.add_argument("--cores", type=int, choices=range(1, _MAX_CORES + 1), default=_DEFAULT_CORES)
    parser.add_argument(
        "--reference-cores",
        type=int,
        choices=range(1, _MAX_CORES + 1),
        help="reference core count for --check (defaults to --cores)",
    )
    parser.add_argument(
        "--reference-variant",
        choices=("legacy-baseline", "baseline", "rank1-gather16", "rank1-gather16-fused"),
        help="reference kernel for --check (defaults to legacy-baseline)",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true", help="run each launch through ACLGraph replay")
    parser.add_argument("--check", action="store_true", help="compare output and state instead of profiling")
    parser.add_argument("--fused-tolerance", action="store_true", help="report FP16 differences using atol/rtol")
    args = parser.parse_args()
    if not 1 <= args.batch <= 1024 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch must be in 1..1024, warmup non-negative and iterations positive")
    if args.check and args.variant == "native":
        parser.error("--check requires a grouped variant")
    diagnostic = args.variant in _DIAGNOSTIC_PROBES
    if args.check and diagnostic:
        parser.error("this diagnostic changes the recurrence and cannot use --check")
    if (args.reference_cores is not None or args.reference_variant is not None) and not args.check:
        parser.error("reference arguments require --check")
    if args.fused_tolerance and (not args.check or args.variant not in _FUSED_PROBES):
        parser.error("--fused-tolerance requires --check with a fused rank-1 variant")
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

    def launch(probe, cores):
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
        op(q, k, v, beta, g, state, slots, accepted, out, scale, False, cores, True, offsets, probe)
        return out

    def prepare(probe):
        if not args.graph:
            return lambda: launch(probe, args.cores)
        state.copy_(initial)
        launch(probe, args.cores)
        torch.npu.synchronize()
        state.copy_(initial)
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            graph_output = launch(probe, args.cores)

        def replay():
            graph.replay()
            return graph_output

        return replay

    probe = _PROBES.get(args.variant, _LEGACY_BASELINE)
    mode = "graph" if args.graph else "eager"
    geometry = f"{args.variant} B{batch} W{window} H{key_heads} {mode}"
    if args.check:
        reference_cores = args.reference_cores or args.cores
        reference = args.reference_variant or "legacy-baseline"
        case = f"{geometry} cores={args.cores} reference={reference} reference_cores={reference_cores}"
        out.fill_(float("nan"))
        launch(_PROBES[reference], reference_cores)
        torch.npu.synchronize()
        expected_out = out.cpu()
        if not expected_out.isfinite().all():
            raise RuntimeError(f"{reference} did not produce finite output")
        expected_state = (initial if args.variant == "no-state-store" else state).cpu()
        run = prepare(probe)
        for repeat in range(1, 4):
            state.copy_(initial)
            out.fill_(float("nan"))
            run()
            torch.npu.synchronize()
            if args.fused_tolerance:
                _assert_bits(state[0], initial[0].cpu(), f"{case} repeat={repeat} reserved slot")
                _check_fused(out, expected_out, f"{case} repeat={repeat} output")
                _check_fused(state, expected_state, f"{case} repeat={repeat} state")
            else:
                _assert_bits(out, expected_out, f"{case} repeat={repeat} output")
                _assert_bits(state, expected_state, f"{case} repeat={repeat} state")
        print(f"PASS {case} repeats=3")
        return

    launch_shape = "native tiling" if native else f"requested blockDim={args.cores}"
    print(f"Profiling {geometry}, {launch_shape}", flush=True)
    run = prepare(probe)
    if diagnostic:
        state.copy_(initial)
        out.fill_(float("nan"))
        run()
        torch.npu.synchronize()
        if not out.cpu().isfinite().all() or not state.cpu().isfinite().all():
            raise RuntimeError(f"{geometry} produced non-finite output or state")
        _assert_bits(state[0], initial[0].cpu(), f"{geometry} reserved slot")
    for _ in range(args.warmup + args.iterations):
        state.copy_(initial)
        run()
    torch.npu.synchronize()
    print(f"Completed {args.warmup + args.iterations} launches of {geometry}, {launch_shape}")


if __name__ == "__main__":
    main()
