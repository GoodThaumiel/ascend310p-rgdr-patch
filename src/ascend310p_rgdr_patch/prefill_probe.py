# SPDX-License-Identifier: Apache-2.0
"""Compare 310P prefill operators across vLLM Ascend images."""

import argparse
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
import time
import traceback
from functools import partial
from importlib import metadata, util
from itertools import accumulate
from pathlib import Path

SUITES = ("env", "topk", "rmsnorm", "gmm", "gdn", "metadata")
SEED = 31024030
CHUNK = 64
CANN_RE = re.compile(r"\d+\.\d+\.\d+(?:[. _-]?(?:beta|rc)\d+)?", re.I)
PATH_VARS = (
    "ASCEND_HOME_PATH",
    "ASCEND_TOOLKIT_HOME",
    "ASCEND_OPP_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "LD_LIBRARY_PATH",
)
LIB_RE = re.compile(
    r"(lib(?:ascendcl|acl_op_compiler|.*opapi|nnopbase|runtime|torch_npu)\.so|vllm_ascend_C.*\.so)"
)


def run_info(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, timeout=8, check=False
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        }
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def loaded_libraries():
    maps = Path("/proc/self/maps")
    if not maps.is_file():
        return []
    paths = set()
    for line in maps.read_text().splitlines():
        fields = line.split(maxsplit=5)
        if (
            len(fields) == 6
            and fields[-1].startswith("/")
            and LIB_RE.search(Path(fields[-1]).name)
        ):
            paths.add(fields[-1])
    return sorted(paths)


def cann_evidence(paths):
    """Read metadata under active paths and the paths of mapped CANN libraries."""
    roots = set()
    for path in filter(None, paths):
        directory = Path(path).resolve()
        if not directory.is_dir():
            directory = directory.parent
        roots.add(directory)
        roots.update(list(directory.parents)[:4])
    records = {}
    for root in sorted(roots):
        for candidate in (
            root / "version.info",
            root / "version.cfg",
            *root.glob("ascend_*_install.info"),
        ):
            if candidate.is_file():
                records[str(candidate)] = candidate.read_text(errors="replace")[:8192]
    return records


def environment(args, torch, torch_npu):
    paths = {name: os.environ.get(name) for name in PATH_VARS}
    libraries = loaded_libraries()
    toolkit_paths = [
        part
        for key in PATH_VARS[:2]
        for part in (paths[key] or "").split(os.pathsep)
        if part
    ]
    active_paths = [
        part
        for value in paths.values()
        for part in (value or "").split(os.pathsep)
        if part
    ]
    runtime_paths = [
        name for name in libraries if Path(name).name.startswith("libascendcl.so")
    ]
    evidence = cann_evidence([*toolkit_paths, *runtime_paths, *active_paths])
    toolkit_evidence = cann_evidence([*toolkit_paths, *runtime_paths])
    modules = {name: util.find_spec(name) for name in ("vllm", "vllm_ascend")}
    return {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "packages": {
            name: package_version(name)
            for name in ("torch", "torch-npu", "vllm", "vllm-ascend")
        },
        "torch_module": torch.__file__,
        "torch_npu_module": torch_npu.__file__,
        "vllm_modules": {
            name: spec.origin if spec else None for name, spec in modules.items()
        },
        "device_index": args.device,
        "device_name": torch.npu.get_device_name(args.device),
        "visible_devices": os.environ.get("ASCEND_RT_VISIBLE_DEVICES"),
        "paths": paths,
        "resolved_toolkit_paths": [str(Path(path).resolve()) for path in toolkit_paths],
        "cann_version_files": evidence,
        "toolkit_version_files": toolkit_evidence,
        "loaded_libraries": libraries,
        "npu_inventory": run_info(["npu-smi", "info", "-l"]),
        "probe_commit": run_info(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"]
        ),
    }


def canonical_version(version):
    return re.sub(r"[. _-](?=(?:beta|rc)\d)", "", version.lower())


def verify_environment(args, env):
    checks = {}
    if args.expect_cann:
        toolkit = env["toolkit_version_files"]
        runtime_paths = [
            path
            for path in env["loaded_libraries"]
            if Path(path).name.startswith("libascendcl.so")
        ]
        evidence = [
            *runtime_paths,
            *env["resolved_toolkit_paths"],
            *toolkit,
            *toolkit.values(),
        ]
        versions = {
            canonical_version(match.group())
            for value in evidence
            for match in CANN_RE.finditer(value)
        }
        expected = canonical_version(args.expect_cann)
        expected_base = expected.split("beta")[0].split("rc")[0]
        runtime_versions = {
            canonical_version(match.group(1))
            for path in runtime_paths
            for match in re.finditer(
                r"/(?:cann[-_])?(" + CANN_RE.pattern + r")(?=/)", path, re.I
            )
        }
        checks["cann"] = {
            "expected": args.expect_cann,
            "observed": sorted(versions),
            "runtime_path_tags": sorted(runtime_versions),
            "passed": expected in versions
            and all(tag in (expected, expected_base) for tag in runtime_versions),
        }
    for name, expected in (
        ("vllm", args.expect_vllm),
        ("vllm-ascend", args.expect_vllm_ascend),
        ("torch-npu", args.expect_torch_npu),
    ):
        if expected:
            actual = env["packages"][name]
            checks[name] = {
                "expected": expected,
                "observed": actual,
                "passed": bool(actual and actual.startswith(expected)),
            }
    if args.expect_device:
        device = env["device_name"]
        checks["device"] = {
            "expected": args.expect_device,
            "observed": device,
            "passed": args.expect_device.lower() in device.lower(),
        }
    return checks


def fingerprint(*values):
    sha = hashlib.sha256()
    for value in values:
        if hasattr(value, "numpy"):
            sha.update(str((tuple(value.shape), value.dtype)).encode())
            sha.update(memoryview(value.contiguous().numpy()).cast("B"))
        else:
            sha.update(json.dumps(value, sort_keys=True).encode())
    return sha.hexdigest()


def tensor_info(tensor, torch_npu):
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "offset": tensor.storage_offset(),
        "format": torch_npu.get_npu_format(tensor),
    }


def representative_output(torch, value):
    tensors = value if isinstance(value, (tuple, list)) else (value,)
    results = []
    for tensor in tensors:
        if not isinstance(tensor, torch.Tensor):
            continue
        rows = tensor.detach().cpu().flatten()
        samples = rows[:: max(1, rows.numel() // 256)][:256].float()
        results.append(
            {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "finite": bool(rows.isfinite().all())
                if tensor.is_floating_point()
                else True,
                "sample": [
                    item if item == item and abs(item) != float("inf") else None
                    for item in samples.tolist()
                ],
            }
        )
    return results


def time_op(torch, operation, warmup, iterations, batch_size):
    for _ in range(warmup):
        operation()
    torch.npu.synchronize()
    # Some 310P images return 107006 from aclrtEventElapsedTime. Use
    # the same host clock and synchronization boundary in both images.
    batch_sizes = []
    times = []
    total_us = 0.0
    for offset in range(0, iterations, batch_size):
        launches = min(batch_size, iterations - offset)
        start_ns = time.perf_counter_ns()
        for _ in range(launches):
            value = operation()
        torch.npu.synchronize()
        elapsed_us = (time.perf_counter_ns() - start_ns) / 1e3
        total_us += elapsed_us
        batch_sizes.append(launches)
        times.append(elapsed_us / launches)
    times.sort()
    return {
        "source": "host_synchronized_batches",
        "sample_batch_sizes": batch_sizes,
        "p50_us": statistics.median(times),
        "p90_us": times[(9 * len(times) + 9) // 10 - 1],
        "min_us": times[0],
        "host_wall_us_per_call": total_us / iterations,
    }, value


class Probe:
    def __init__(self, args, torch, torch_npu, report):
        self.args = args
        self.torch = torch
        self.torch_npu = torch_npu
        self.report = report
        self.device = torch.device(f"npu:{args.device}")

    def save(self):
        path = self.args.output
        temp = path.with_suffix(path.suffix + ".tmp")
        temp.write_text(json.dumps(self.report, indent=2, allow_nan=False) + "\n")
        temp.replace(path)

    def case(
        self,
        name,
        operation=None,
        *,
        fixture=None,
        inputs=(),
        check=None,
        reason=None,
        **info,
    ):
        if self.args.case and not re.search(self.args.case, name):
            return
        row = {"name": name, "status": "RUNNING", **info}
        self.report["cases"].append(row)
        print(f"[prefill-probe] {name}", flush=True)
        self.save()
        if operation is None:
            row.update(status="SKIPPED", reason=reason)
        else:
            try:
                row["fixture_sha256"] = fixture
                row["inputs"] = [tensor_info(item, self.torch_npu) for item in inputs]
                timing, output = time_op(
                    self.torch,
                    operation,
                    self.args.warmup,
                    self.args.iterations,
                    self.args.timing_batch_size,
                )
                row.update(
                    timing=timing, output=representative_output(self.torch, output)
                )
                if not all(item["finite"] for item in row["output"]):
                    raise ValueError("non-finite tensor output")
                if check is not None:
                    row["check"] = check(output)
                row["status"] = "PASSED"
                print(f"  p50={timing['p50_us']:.3f} us", flush=True)
            except Exception as error:
                row.update(
                    status="FAILED", error=str(error), traceback=traceback.format_exc()
                )
                first_line = str(error).partition("\n")[0]
                print(f"  FAILED: {first_line}", flush=True)
        self.save()

    def topk(self):
        torch, npu = self.torch, self.torch_npu
        for rows in (816, 929, 1024):
            rng = torch.Generator().manual_seed(SEED + rows)
            cpu = torch.randn((rows, 256), generator=rng).half()
            logits = cpu.to(self.device)
            fixture = fingerprint(cpu, 8)

            def native():
                return npu.npu_moe_gating_top_k_softmax(logits, k=8)

            def native_renorm():
                weights, ids, _ = native()
                return weights / weights.sum(dim=-1, keepdim=True), ids.int()

            def selected():
                values, ids = logits.topk(8, dim=-1)
                return values.float().softmax(dim=-1).half(), ids.int()

            def split512():
                pieces = [
                    npu.npu_moe_gating_top_k_softmax(piece, k=8)
                    for piece in logits.split(512)
                ]
                weights = torch.cat([item[0] for item in pieces])
                ids = torch.cat([item[1] for item in pieces])
                return weights / weights.sum(dim=-1, keepdim=True), ids.int()

            # Check each candidate against exact CPU logits, including tied FP16 scores.
            target, _ = cpu.float().topk(8, dim=-1)

            def check(output, renorm):
                weights, ids = output[:2]
                scores = cpu.float().gather(-1, ids.cpu().long())
                weights = weights.cpu().float()
                expected = (
                    scores.softmax(dim=-1)
                    if renorm
                    else ((scores - cpu.float().logsumexp(dim=-1, keepdim=True)).exp())
                )
                duplicates = (
                    ids.cpu().sort(dim=-1).values.diff(dim=-1).eq(0).any(dim=-1)
                )
                return {
                    "rows_with_wrong_scores": int(
                        (scores.sort(dim=-1).values != target.flip((-1,)))
                        .any(dim=-1)
                        .sum()
                    ),
                    "rows_with_duplicate_ids": int(duplicates.sum()),
                    "weight_max_abs": (weights - expected).abs().max().item(),
                }

            for name, op in (
                ("native", native),
                ("native_renorm", native_renorm),
                ("selected_softmax", selected),
                ("split512_renorm", split512),
            ):
                self.case(
                    f"topk/{rows}x256/{name}",
                    op,
                    fixture=fixture,
                    inputs=(logits,),
                    check=partial(check, renorm=name != "native"),
                )

    def rmsnorm(self):
        torch, npu = self.torch, self.torch_npu
        for rows, width in (
            (624, 128),
            (768, 128),
            (30480, 128),
            (31232, 128),
            (31240, 128),
            (31248, 128),
            (31264, 128),
            (62208, 128),
            (64016, 128),
            (3888, 2048),
            (4001, 2048),
        ):
            rng = torch.Generator().manual_seed(SEED + rows)
            cpu = torch.randn((rows, width), generator=rng).half()
            gamma_cpu = torch.full((width,), width**-0.5, dtype=torch.float16)
            x, gamma = cpu.to(self.device), gamma_cpu.to(self.device)
            eps = 1e-6 / width
            fixture = fingerprint(cpu, gamma_cpu, eps)
            ids = list(range(0, rows, max(1, rows // 256)))[:256]
            sample = cpu[ids].float()
            reference = (
                sample
                * sample.square().mean(dim=-1, keepdim=True).add(eps).rsqrt()
                * gamma_cpu.float()
            ).half()

            def check(output):
                got = output[0].cpu()[ids]
                return {
                    "sample_max_abs": (got.float() - reference.float())
                    .abs()
                    .max()
                    .item()
                }

            def native(tensor):
                return npu.npu_rms_norm(tensor, gamma, eps)

            self.case(
                f"rmsnorm/{rows}x{width}/native",
                partial(native, x),
                fixture=fixture,
                inputs=(x, gamma),
                check=check,
            )
            if rows == 31240:
                for multiple in (16, 32):

                    def padded():
                        pad = x.new_zeros((multiple - rows % multiple, width))
                        y, rstd = native(torch.cat((x, pad), dim=0))
                        return y[:rows], rstd[:rows]

                    self.case(
                        f"rmsnorm/{rows}x{width}/pad{multiple}",
                        padded,
                        fixture=fixture,
                        inputs=(x, gamma),
                        check=check,
                        padded_rows=rows + multiple - rows % multiple,
                    )

    def gmm(self):
        torch, npu = self.torch, self.torch_npu
        hidden, local_intermediate, experts = (
            2048,
            self.args.gmm_intermediate,
            self.args.gmm_experts,
        )
        for name, input_dim, output_dim in (
            ("gmm1", hidden, 2 * local_intermediate),
            ("gmm2", local_intermediate, hidden),
        ):
            rng = torch.Generator().manual_seed(SEED + input_dim)
            cpu_weight = torch.randint(
                -16,
                17,
                (experts, output_dim, input_dim),
                generator=rng,
                dtype=torch.int8,
            )
            cpu_scale = (
                torch.rand((experts, output_dim), generator=rng) * 0.01 + 0.005
            ).float()
            weight_key = fingerprint(cpu_weight, cpu_scale)
            weight = npu.npu_format_cast(cpu_weight.to(self.device), 29)  # FRACTAL_NZ
            scale = cpu_scale.to(self.device)
            del cpu_weight
            for tokens in self.args.gmm_tokens:
                rows = tokens * 8
                cpu_x = (torch.randn((rows, input_dim), generator=rng) * 0.2).half()
                x = cpu_x.to(self.device)
                for pattern in ("balanced", "skewed"):
                    if pattern == "balanced":
                        counts = [
                            rows // experts + (i < rows % experts)
                            for i in range(experts)
                        ]
                    else:
                        offsets = list(
                            accumulate([i % 32 + 1 for i in range(experts)], initial=0)
                        )
                        counts = [
                            right * rows // offsets[-1] - left * rows // offsets[-1]
                            for left, right in zip(offsets, offsets[1:])
                        ]
                    cpu_group = torch.tensor(
                        list(accumulate(counts)), dtype=torch.int64
                    )
                    group = cpu_group.to(self.device)
                    op = partial(
                        npu.npu_quant_grouped_matmul_dequant,
                        x=x,
                        quantized_weight=weight,
                        weight_scale=scale,
                        group_list=group,
                        quant_mode="pertoken",
                    )
                    self.case(
                        f"gmm/{tokens}/{pattern}/{name}",
                        op,
                        fixture=fingerprint(weight_key, cpu_x, cpu_group),
                        inputs=(x, weight, scale, group),
                        expert_counts=counts,
                    )
            del weight, scale

    def gdn(self):
        torch = self.torch
        (
            q,
            k,
            v,
            g,
            beta,
            q_kernel,
            k_kernel,
            w,
            u,
            g_kernel,
            state,
            o_v,
            o_h,
            cu,
            chunks,
        ) = gdn_fixture(torch, self.args.gdn_lengths)
        wy_key = fingerprint(q, k, v, g, beta, CHUNK)
        h_key = fingerprint(k_kernel, w, u, g_kernel, state, cu, chunks)
        o_key = fingerprint(q_kernel, k_kernel, o_v, o_h, g_kernel, cu, chunks)
        kernel_data = (q_kernel, k_kernel, w, u, g_kernel, state, o_v, o_h)
        q_dev, k_dev, w_dev, u_dev, g_dev, state_dev, o_v_dev, o_h_dev = [
            item.to(self.device) for item in kernel_data
        ]
        shape_info = {
            "valid_lengths": self.args.gdn_lengths,
            "padded_offsets": cu,
            "chunk_indices": chunks,
        }
        runtime = torch.ops._C_ascend
        for variant, suffix in (("legacy", ""), ("vllm", "_vllm")):
            h_name, o_name = (
                f"chunk_gated_delta_rule_fwd_h{suffix}",
                f"chunk_fwd_o{suffix}",
            )
            h_op, o_op = getattr(runtime, h_name, None), getattr(runtime, o_name, None)
            h_inputs = (k_dev, w_dev, u_dev, g_dev, state_dev)
            o_inputs = (q_dev, k_dev, o_v_dev, o_h_dev, g_dev)
            h = (
                None
                if h_op is None
                else partial(
                    h_op,
                    k_dev,
                    w_dev,
                    u_dev,
                    g=g_dev,
                    gk=None,
                    initial_state=state_dev,
                    output_final_state=True,
                    chunk_size=CHUNK,
                    save_new_value=True,
                    cu_seqlens=cu,
                    chunk_indices=chunks,
                    use_exp2=False,
                    transpose_state_layout=False,
                )
            )

            def run_o(values, states):
                return o_op(
                    q_dev,
                    k_dev,
                    values,
                    states,
                    128**-0.5,
                    g=g_dev,
                    g_gamma=None,
                    cu_seqlens=cu,
                    chunk_indices=chunks,
                    chunk_size=CHUNK,
                    transpose_state_layout=False,
                )

            def chain():
                h_out, values, final_state = h()
                return run_o(values, h_out), final_state

            self.case(
                f"gdn/h/{variant}",
                h,
                fixture=h_key,
                inputs=h_inputs,
                comparison_key="gdn/h",
                reason=f"_C_ascend.{h_name} unavailable",
                binding=h_name,
                **shape_info,
            )
            self.case(
                f"gdn/o/{variant}",
                partial(run_o, o_v_dev, o_h_dev) if o_op is not None else None,
                fixture=o_key,
                inputs=o_inputs,
                comparison_key="gdn/o",
                reason=f"_C_ascend.{o_name} unavailable",
                binding=o_name,
                **shape_info,
            )
            self.case(
                f"gdn/chain/{variant}",
                chain if h is not None and o_op is not None else None,
                fixture=fingerprint(h_key, q_kernel),
                inputs=(q_dev, *h_inputs),
                comparison_key="gdn/chain",
                reason=f"requires {h_name} and {o_name}",
                bindings=[h_name, o_name],
                **shape_info,
            )

        wy_inputs = tuple(item.to(self.device) for item in (q, k, v, g, beta))
        for variant, wy_op in (
            ("legacy", getattr(runtime, "chunk_gated_delta_rule_compute_wy", None)),
            (
                "standalone",
                getattr(torch.ops.replayssm_310p_compute_wy, "prepare", None),
            ),
        ):
            self.case(
                f"gdn/wy/{variant}",
                partial(wy_op, *wy_inputs, CHUNK) if wy_op is not None else None,
                fixture=wy_key,
                inputs=wy_inputs,
                comparison_key="gdn/wy",
                reason="WY binding unavailable",
                **shape_info,
            )

    def metadata(self):
        torch = self.torch
        offsets = list(accumulate(self.args.gdn_lengths, initial=0))
        cu_dev = torch.tensor(offsets, dtype=torch.int32, device=self.device)
        offsets_host = tuple(offsets)

        def from_device():
            return tuple(cu_dev.to(torch.int64).cpu().tolist())

        for name, op in (
            ("device_to_host", from_device),
            ("cached_host", lambda: offsets_host),
        ):
            if self.args.case and not re.search(self.args.case, f"metadata/{name}"):
                continue
            for _ in range(self.args.warmup):
                op()
            torch.npu.synchronize()
            times = []
            for _ in range(self.args.iterations):
                start = time.perf_counter()
                result = op()
                times.append((time.perf_counter() - start) * 1e6)
                if result != offsets_host:
                    raise ValueError(f"unexpected sequence offsets: {result}")
            times.sort()
            self.report["cases"].append(
                {
                    "name": f"metadata/{name}",
                    "status": "PASSED",
                    "fixture_sha256": fingerprint(offsets),
                    "offsets": offsets,
                    "timing": {
                        "source": "host_wall",
                        "p50_us": statistics.median(times),
                        "p90_us": times[(9 * len(times) + 9) // 10 - 1],
                        "min_us": times[0],
                    },
                }
            )
            self.save()


def gdn_fixture(torch, lengths):
    """Generate shared BTHD WY inputs and CPU-derived BHTD inputs for H/O."""
    heads, value_heads, dim = 8, 16, 128
    padded = [((length + CHUNK - 1) // CHUNK) * CHUNK for length in lengths]
    cu = list(accumulate(padded, initial=0))
    chunks = [
        index
        for seq, length in enumerate(padded)
        for i in range(length // CHUNK)
        for index in (seq, i)
    ]
    tokens, n_chunks = cu[-1], cu[-1] // CHUNK
    rng = torch.Generator().manual_seed(SEED)

    def normal(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).half()

    q = torch.nn.functional.normalize(
        normal((1, tokens, heads, dim), 1).float(), dim=-1
    ).half()
    k = torch.nn.functional.normalize(normal(q.shape, 1).float(), dim=-1).half()
    v = normal((1, tokens, value_heads, dim), 0.05)
    g = -torch.rand((1, tokens, value_heads), generator=rng) * 0.03
    beta = (torch.rand(g.shape, generator=rng) * 0.3 + 0.2).half()
    for start, end, valid in zip(cu[:-1], cu[1:], lengths):
        for tensor in (q, k, v, g, beta):
            tensor[:, start + valid : end].zero_()

    q_kernel, k_kernel = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous()
    shape = (1, value_heads, n_chunks, CHUNK)
    key = (
        k_kernel.repeat_interleave(value_heads // heads, dim=1)
        .float()
        .reshape(*shape, dim)
    )
    value = v.transpose(1, 2).float().reshape(*shape, dim)
    decay = g.transpose(1, 2).reshape(shape).cumsum(dim=-1)
    beta_chunk = beta.transpose(1, 2).float().reshape(*shape, 1)
    k_beta = key * beta_chunk
    lower = (
        (k_beta @ key.transpose(-1, -2))
        * (decay.unsqueeze(-1) - decay.unsqueeze(-2)).exp()
    ).tril(-1)
    rhs = torch.cat((k_beta * decay.exp().unsqueeze(-1), value * beta_chunk), dim=-1)
    solved = torch.linalg.solve_triangular(lower, rhs, upper=False, unitriangular=True)
    w, u = (
        part.reshape(1, value_heads, tokens, dim).half().contiguous()
        for part in solved.split(dim, dim=-1)
    )
    g_kernel = decay.reshape(1, value_heads, tokens).contiguous()
    state = normal((len(lengths), value_heads, dim, dim), 0.02)
    # Fixed O-only inputs decouple FwdO timing from any differences in FwdH output.
    o_v = normal(u.shape, 0.02)
    o_h = normal((1, value_heads, n_chunks, dim, dim), 0.02)
    return (
        q,
        k,
        v,
        g,
        beta,
        q_kernel,
        k_kernel,
        w,
        u,
        g_kernel,
        state,
        o_v,
        o_h,
        cu,
        chunks,
    )


def compare_reports(cases, reference):
    comparisons = []
    for row in cases:
        if row["status"] != "PASSED":
            continue
        key = row.get("comparison_key", row["name"])
        candidates = [
            old
            for old in reference["cases"]
            if old["status"] == "PASSED"
            and old.get("comparison_key", old["name"]) == key
        ]
        for old in candidates:
            same_fixture = row["fixture_sha256"] == old["fixture_sha256"]
            current_source = row["timing"].get(
                "source", row.get("timing_source", "npu_event")
            )
            reference_source = old["timing"].get(
                "source", old.get("timing_source", "npu_event")
            )
            same_timing_method = current_source == reference_source and row[
                "timing"
            ].get("sample_batch_sizes") == old["timing"].get("sample_batch_sizes")
            result = {
                "current": row["name"],
                "reference": old["name"],
                "same_fixture": same_fixture,
                "same_timing_method": same_timing_method,
            }
            before = old["timing"]["p50_us"]
            if same_fixture and same_timing_method and before > 0:
                result.update(
                    reference_p50_us=before,
                    current_p50_us=row["timing"]["p50_us"],
                    p50_ratio=row["timing"]["p50_us"] / before,
                )
            if same_fixture:
                differences = []
                for actual, expected in zip(
                    row.get("output", []), old.get("output", [])
                ):
                    if actual["shape"] != expected["shape"] or not (
                        actual["finite"] and expected["finite"]
                    ):
                        differences.append(None)
                        continue
                    differences.append(
                        max(
                            (
                                abs(a - b)
                                for a, b in zip(actual["sample"], expected["sample"])
                            ),
                            default=0.0,
                        )
                    )
                result["output_sample_max_abs"] = differences
            comparisons.append(result)
    return comparisons


def int_list(value):
    try:
        values = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected comma-separated integers") from error
    if any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("all lengths must be positive")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        action="append",
        choices=SUITES,
        help="repeatable; default runs all suites",
    )
    parser.add_argument(
        "--case",
        help="regular expression matched against case names, for focused msprof runs",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--label", default="")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument(
        "--timing-batch-size",
        type=int,
        default=5,
        help="launches per synchronized host timing sample",
    )
    parser.add_argument("--expect-cann")
    parser.add_argument("--expect-vllm")
    parser.add_argument("--expect-vllm-ascend")
    parser.add_argument("--expect-torch-npu")
    parser.add_argument(
        "--expect-device",
        help="case-insensitive substring of torch.npu.get_device_name",
    )
    parser.add_argument("--gmm-tokens", type=int_list, default=(3888, 4001))
    parser.add_argument("--gmm-experts", type=int, default=256)
    parser.add_argument(
        "--gmm-intermediate",
        type=int,
        default=256,
        help="local post-SwiGLU width on one rank",
    )
    parser.add_argument("--gdn-lengths", type=int_list, default=(1024, 896, 1280, 896))
    parser.add_argument(
        "--extra-library",
        action="append",
        type=Path,
        default=[],
        help="additional Torch op library",
    )
    args = parser.parse_args(argv)
    if (
        args.device < 0
        or args.warmup < 0
        or args.iterations < 1
        or args.timing_batch_size < 1
        or args.gmm_experts < 1
        or args.gmm_intermediate < 1
    ):
        parser.error(
            "device and warmup must be nonnegative; iterations, timing batch size, expert count and intermediate width must be positive"
        )
    if args.expect_cann and not CANN_RE.fullmatch(args.expect_cann):
        parser.error("--expect-cann must be a version such as 9.1.0-beta1 or 9.2.1")
    if args.case:
        try:
            re.compile(args.case)
        except re.error as error:
            parser.error(f"invalid --case expression: {error}")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    suites = args.suite or list(SUITES)
    report = {
        "schema": 2,
        "label": args.label,
        "suites": suites,
        "seed": SEED,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing_batch_size": args.timing_batch_size,
        "environment": {},
        "environment_checks": {},
        "suite_errors": {},
        "cases": [],
    }
    try:
        import torch
        import torch_npu

        if "gdn" in suites:
            from vllm_ascend.utils import bootstrap_custom_op_env

            bootstrap_custom_op_env()
        torch.set_num_threads(1)
        torch.npu.set_device(args.device)
        torch_npu.npu.set_compile_mode(jit_compile=False)
        torch.npu.synchronize()
        probe = Probe(args, torch, torch_npu, report)
        report["environment"] = environment(args, torch, torch_npu)
        report["environment_checks"] = verify_environment(args, report["environment"])
        probe.save()
        if not all(check["passed"] for check in report["environment_checks"].values()):
            raise RuntimeError("environment validation failed; see environment_checks")
        if "gdn" in suites:
            try:
                from vllm_ascend.utils import enable_custom_op

                report["environment"]["vllm_ascend_custom_ops"] = bool(
                    enable_custom_op()
                )
                libraries = list(args.extra_library)
                wy_library = os.getenv("VLLM_ASCEND_310P_COMPUTE_WY_LIBRARY")
                if wy_library:
                    libraries.append(Path(wy_library))
                for library in dict.fromkeys(
                    path.resolve(strict=True) for path in libraries
                ):
                    torch.ops.load_library(str(library))
            except Exception:
                report["suite_errors"]["gdn_library_init"] = traceback.format_exc()

        with torch.inference_mode():
            for suite in dict.fromkeys(suites):
                if suite == "env":
                    continue
                try:
                    getattr(probe, suite)()
                except Exception:
                    report["suite_errors"][suite] = traceback.format_exc()
                    probe.save()
        if (
            args.case
            and not report["cases"]
            and any(suite != "env" for suite in suites)
        ):
            raise RuntimeError("--case filtered out every case")
        if not args.case and "gdn" in suites:
            for key in ("gdn/h", "gdn/o", "gdn/wy"):
                if not any(
                    row.get("comparison_key") == key and row["status"] == "PASSED"
                    for row in report["cases"]
                ):
                    report["suite_errors"][key] = (
                        "no working binding for this comparison"
                    )
        report["environment"]["loaded_libraries_after"] = loaded_libraries()
        if args.reference_json:
            ref = json.loads(args.reference_json.read_text())
            report["comparison"] = compare_reports(report["cases"], ref)

        failed = bool(report["suite_errors"]) or any(
            row["status"] == "FAILED" for row in report["cases"]
        )
        report["status"] = "FAILED" if failed else "PASSED"
        probe.save()
    except Exception as error:
        report.update(
            status="FAILED", error=str(error), traceback=traceback.format_exc()
        )
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']} {args.output}", flush=True)
    return int(report["status"] != "PASSED")


if __name__ == "__main__":
    raise SystemExit(main())
