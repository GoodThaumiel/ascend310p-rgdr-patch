# SPDX-License-Identifier: Apache-2.0
"""Check and profile the 310P3 UB-to-L0C broadcast and Mmad scheduling."""

import argparse

import torch

from ascend310p_rgdr_patch.profile import _assert_bits

_MODES = {"gather": 0, "broadcast-raw": 1, "broadcast": 2, "mmad": 3, "serial": 4, "interleaved": 5}
_MMAD_MODES = {"mmad", "serial", "interleaved"}


def _benchmark(args, device):
    delta = (torch.arange(128, dtype=torch.float16) - 64) / 64
    offsets = (torch.arange(128, dtype=torch.int32) * 2).repeat_interleave(16)
    lhs = torch.full((128 * 144,), 0.125, dtype=torch.float16, device=device)
    rhs = torch.full((144 * 16,), 0.0625, dtype=torch.float16, device=device)
    expanded = torch.full((128, 16), float("nan"), dtype=torch.float16, device=device)
    matrix = torch.full_like(expanded, float("nan"))
    inputs = delta.to(device), offsets.to(device), lhs, rhs
    mode = _MODES[args.variant]

    def launch():
        torch.ops.rgdr_v310_l0c_probe.bench_(*inputs, expanded, matrix, mode, args.repeats)

    run = launch
    if args.graph:
        launch()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            launch()
        run = graph.replay
    expanded.fill_(float("nan"))
    matrix.fill_(float("nan"))
    run()
    torch.npu.synchronize()
    if args.variant == "mmad":
        assert expanded.cpu().isnan().all(), "Mmad-only modified expanded delta"
    else:
        _assert_bits(expanded, delta[:, None].expand(128, 16), f"{args.variant} expanded delta")
    if args.variant in _MMAD_MODES:
        _assert_bits(matrix, torch.full((128, 16), 1.125, dtype=torch.float16), f"{args.variant} Mmad output")
    else:
        assert matrix.cpu().isnan().all(), "broadcast-only modified Mmad output"
    print(f"PASS {args.variant} repeats={args.repeats} blockDim=1", flush=True)
    if not args.check:
        for _ in range(args.warmup + args.iterations):
            run()
        torch.npu.synchronize()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--variant", choices=("layout", *_MODES), default="layout")
    parser.add_argument("--repeats", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--graph", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.repeats <= 256 or args.warmup < 0 or args.iterations < 1:
        parser.error("repeats must be in [1,256], warmup non-negative and iterations positive")

    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(args.library)
    if args.variant != "layout":
        _benchmark(args, device)
        return
    values = (torch.arange(16, dtype=torch.float16) - 8) * 0.25
    source = values.to(device)
    out = torch.full((16, 16), float("nan"), dtype=torch.float16, device=device)
    torch.ops.rgdr_v310_l0c_probe.run_(source, out)
    torch.npu.synchronize()

    result = out.cpu()
    print(f"first row: {result[0].tolist()}")
    print(f"first column: {result[:, 0].tolist()}")
    layout = {
        "rows": values[:, None].expand_as(result),
        "columns": values[None, :].expand_as(result),
    }
    for name, expected in layout.items():
        if torch.equal(result.view(torch.int16), expected.view(torch.int16)):
            print(f"PASS: each input element occupies one complete {name[:-1]}")
            return
    raise RuntimeError(f"Unexpected UB-to-L0C tile mapping: {result.tolist()}")


if __name__ == "__main__":
    main()
