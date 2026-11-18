# SPDX-License-Identifier: Apache-2.0
"""Determine the FP16 tile layout of the 310P3 UB-to-L0C broadcast."""

import argparse

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()

    import torch_npu

    device = torch.device(f"npu:{args.device}")
    torch.npu.set_device(device)
    torch_npu.npu.set_compile_mode(jit_compile=False)
    torch.ops.load_library(args.library)
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
