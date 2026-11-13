# SPDX-License-Identifier: Apache-2.0
"""Opt-in replacement of the 0.23.0 Ascend310P GDN recurrent helper."""

import inspect
import logging
import os
import re
from functools import wraps
from importlib import import_module
from importlib.metadata import version
from pathlib import Path

_LOG = logging.getLogger(__name__)
_LIBRARY_ENV = "ASCEND310P_RGDR_PATCH_LIBRARY"
_MODE_ENV = "ASCEND310P_RGDR_PATCH_MODE"
_GDN_MODULE = "vllm_ascend._310p.ops.fla.gdn_310"
_HELPER = "npu_recurrent_gated_delta_rule_310"
_PARAMETERS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "state",
    "cu_seqlens",
    "ssm_state_indices",
    "num_accepted_tokens",
    "use_qk_l2norm_in_kernel",
)
_KEY_HEADS = 8
_VALUE_HEADS = 16
_HEAD_DIM = 128
_STATE_ELEMS = _VALUE_HEADS * _HEAD_DIM * _HEAD_DIM
_MAX_BATCH = 1024
_MAX_WINDOW = 8
_CORES = 7


def _matches_geometry(q, k, v, g, beta, state, offsets, slots, accepted, torch):
    if offsets is None or offsets.ndim != 1 or slots is None or g is None:
        return False
    batch = offsets.numel() - 1
    if not 1 <= batch <= _MAX_BATCH or v.ndim != 4:
        return False
    tokens = v.shape[1]
    if not 1 <= tokens <= _MAX_BATCH * _MAX_WINDOW:
        return False
    if accepted is None:
        if slots.ndim != 1 or slots.numel() < batch:
            return False
    elif (
        slots.ndim != 2
        or slots.shape[0] < batch
        or not 1 <= slots.shape[1] <= _MAX_WINDOW
        or accepted.ndim != 1
        or accepted.numel() < batch
    ):
        return False
    return (
        q.shape == k.shape == (1, tokens, _KEY_HEADS, _HEAD_DIM)
        and v.shape == (1, tokens, _VALUE_HEADS, _HEAD_DIM)
        and g.shape == beta.shape == (1, tokens, _VALUE_HEADS)
        and state.ndim == 4
        and state.shape[0] > 1
        and state.shape[1:] == (_VALUE_HEADS, _HEAD_DIM, _HEAD_DIM)
        and state.dtype == torch.float16
        and state.stride()[1:] == (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
        and state.stride(0) >= _STATE_ELEMS
        and state.stride(0) % 16 == 0
        and state.storage_offset() % 16 == 0
        and all(t.device == v.device for t in (q, k, g, beta, state, offsets, slots))
        and (accepted is None or accepted.device == v.device)
    )


def _wrap(original, normalize, grouped, mode, torch):
    @wraps(original)
    def replacement(
        q,
        k,
        v,
        g,
        beta,
        state,
        cu_seqlens,
        ssm_state_indices,
        num_accepted_tokens=None,
        use_qk_l2norm_in_kernel=True,
    ):
        speculative = num_accepted_tokens is not None
        if (mode != "both" and speculative != (mode == "spec")) or not _matches_geometry(
            q,
            k,
            v,
            g,
            beta,
            state,
            cu_seqlens,
            ssm_state_indices,
            num_accepted_tokens,
            torch,
        ):
            return original(
                q,
                k,
                v,
                g,
                beta,
                state,
                cu_seqlens,
                ssm_state_indices,
                num_accepted_tokens,
                use_qk_l2norm_in_kernel,
            )
        if use_qk_l2norm_in_kernel:
            q = normalize(q)
            k = normalize(k)

        offsets = cu_seqlens.to(torch.int32).contiguous()
        batch = offsets.numel() - 1
        slots = ssm_state_indices[:batch].to(torch.int32).contiguous()
        if speculative:
            accepted = num_accepted_tokens[:batch].to(torch.int32).contiguous()
        else:
            # The current slot is both the initial state and the sole writeback.
            # Zero-length graph padding rows are skipped by the packed kernel.
            slots = slots.unsqueeze(1)
            accepted = torch.ones_like(offsets[1:])

        out = torch.empty_like(v, dtype=torch.float16, memory_format=torch.contiguous_format)
        grouped(
            q.squeeze(0).to(torch.float16).contiguous(),
            k.squeeze(0).to(torch.float16).contiguous(),
            v.squeeze(0).to(torch.float16).contiguous(),
            beta.squeeze(0).to(torch.float16).contiguous(),
            g.squeeze(0).to(torch.float16).contiguous(),
            state,
            slots,
            accepted,
            out.squeeze(0),
            _HEAD_DIM**-0.5,
            False,
            _CORES,
            True,
            offsets,
        )
        return out

    replacement._ascend310p_grouped_rgdr = True
    return replacement


def register():
    """Called by vLLM's general plugin loader in driver and worker processes."""
    library = os.getenv(_LIBRARY_ENV)
    if not library:
        return
    mode = os.getenv(_MODE_ENV, "decode")
    if mode not in {"decode", "spec", "both"}:
        raise ValueError(f"{_MODE_ENV} must be decode, spec or both")
    for package in ("vllm", "vllm-ascend"):
        installed = version(package)
        if not re.match(r"^0\.23\.0(?:$|[.+-])", installed):
            raise RuntimeError(f"Grouped RGDR patch expects {package} 0.23.0, found {installed}")
    path = Path(library).expanduser()
    if not path.is_absolute() or not path.is_file():
        raise ValueError(f"{_LIBRARY_ENV} must point to an absolute path to the built .so")

    from vllm_ascend.utils import is_310p

    if not is_310p():
        raise RuntimeError("Grouped RGDR patch requires the 310P build of vLLM Ascend")
    gdn = import_module(_GDN_MODULE)
    original = getattr(gdn, _HELPER)
    if getattr(original, "_ascend310p_grouped_rgdr", False):
        return
    if tuple(inspect.signature(original).parameters) != _PARAMETERS:
        raise RuntimeError(f"Unexpected signature for {_GDN_MODULE}.{_HELPER}")

    import torch

    torch.ops.load_library(str(path))
    op = getattr(torch.ops.rgdr_v310_cube_fp16, "run_", None)
    if op is None or not any(arg.name == "row_offsets" for arg in op.default._schema.arguments):
        raise RuntimeError("The RGDR .so must export run_ with packed row_offsets support")
    setattr(gdn, _HELPER, _wrap(original, gdn.l2norm_310p, op.default, mode, torch))
    _LOG.info("310P grouped RGDR monkey patch installed (mode=%s, library=%s)", mode, path)
