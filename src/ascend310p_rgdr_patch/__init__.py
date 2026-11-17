# SPDX-License-Identifier: Apache-2.0
"""Opt-in replacement of the 0.23.0/0.23.0rc1 Ascend310P GDN recurrent helper."""

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
_CORES_ENV = "ASCEND310P_RGDR_PATCH_CORES"
_DIAGNOSTICS_ENV = "ASCEND310P_RGDR_PATCH_DIAGNOSTICS"
_SUPPORTED_VERSION = re.compile(r"0\.23\.0(?:rc1)?(?:[.+-].+)?")
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
_KEY_HEAD_COUNTS = (8, 16)
_HEAD_DIM = 128
_MAX_BATCH = 1024
_MAX_WINDOW = 8
_DEFAULT_CORES = 7
_MAX_CORES = 8


def _geometry_mismatch(q, k, v, g, beta, state, offsets, slots, accepted, torch):
    if offsets is None or offsets.ndim != 1 or slots is None or g is None:
        return "missing or incompatible offsets, slots, or gate"
    batch = offsets.numel() - 1
    if not 1 <= batch <= _MAX_BATCH or v.ndim != 4:
        return "unsupported batch size or V rank"
    tokens = v.shape[1]
    if not 1 <= tokens <= _MAX_BATCH * _MAX_WINDOW:
        return "unsupported token count"
    if accepted is None:
        if slots.ndim != 1 or slots.numel() < batch:
            return "incompatible decode slots"
    elif (
        slots.ndim != 2
        or slots.shape[0] < batch
        or not 1 <= slots.shape[1] <= _MAX_WINDOW
        or accepted.ndim != 1
        or accepted.numel() < batch
    ):
        return "incompatible speculative slots or accepted-token metadata"
    key_heads = q.shape[2] if q.ndim == 4 else 0
    value_heads = 2 * key_heads
    if (
        key_heads not in _KEY_HEAD_COUNTS
        or q.shape != (1, tokens, key_heads, _HEAD_DIM)
        or k.shape != q.shape
        or v.shape != (1, tokens, value_heads, _HEAD_DIM)
        or g.shape != (1, tokens, value_heads)
        or beta.shape != g.shape
    ):
        return "unsupported Q/K/V or gate/beta shape"
    if not (state.ndim == 4 and state.shape[0] > 1 and state.shape[1:] == (value_heads, _HEAD_DIM, _HEAD_DIM)):
        return "unsupported state shape"
    if state.dtype != torch.float16:
        return "state must be FP16"
    if not (
        state.layout == torch.strided
        and state.stride()[1:] == (_HEAD_DIM * _HEAD_DIM, _HEAD_DIM, 1)
        and state.stride(0) >= value_heads * _HEAD_DIM * _HEAD_DIM
        and state.stride(0) % 16 == 0
        and state.storage_offset() % 16 == 0
    ):
        return "unsupported state strides or alignment"
    if any(t.device != v.device for t in (q, k, g, beta, state, offsets, slots)) or (
        accepted is not None and accepted.device != v.device
    ):
        return "inputs or metadata are on different devices"
    return None


def _tensor_metadata(tensor):
    if tensor is None:
        return None
    return (tuple(tensor.shape), str(tensor.dtype), str(tensor.device))


def _wrap(original, normalize, grouped, mode, torch, *, cores=_DEFAULT_CORES, diagnostics=False, is_capturing=None):
    logged_routes = set()

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
        if mode != "both" and speculative != (mode == "spec"):
            reason = "configured mode excludes this call"
        else:
            reason = _geometry_mismatch(
                q, k, v, g, beta, state, cu_seqlens, ssm_state_indices, num_accepted_tokens, torch
            )
        capturing = diagnostics and is_capturing is not None and is_capturing()
        if diagnostics and (speculative, reason, capturing) not in logged_routes:
            logged_routes.add((speculative, reason, capturing))
            _LOG.warning(
                "RGDR dispatch pid=%s mode=%s speculative=%s route=%s reason=%s "
                "q=%s k=%s v=%s gate=%s beta=%s state=%s state_stride=%s state_offset=%s "
                "offsets=%s slots=%s accepted=%s capturing=%s",
                os.getpid(),
                mode,
                speculative,
                "native" if reason else "grouped",
                reason or "matched",
                _tensor_metadata(q),
                _tensor_metadata(k),
                _tensor_metadata(v),
                _tensor_metadata(g),
                _tensor_metadata(beta),
                _tensor_metadata(state),
                state.stride() if state.layout == torch.strided else None,
                state.storage_offset() if state.layout == torch.strided else None,
                _tensor_metadata(cu_seqlens),
                _tensor_metadata(ssm_state_indices),
                _tensor_metadata(num_accepted_tokens),
                capturing,
            )
        if reason:
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
            cores,
            True,
            offsets,
        )
        return out

    replacement._ascend310p_grouped_rgdr = True
    return replacement


def _configured_cores():
    try:
        cores = int(os.getenv(_CORES_ENV, str(_DEFAULT_CORES)))
    except ValueError as error:
        raise ValueError(f"{_CORES_ENV} must be an integer in [1, {_MAX_CORES}]") from error
    if not 1 <= cores <= _MAX_CORES:
        raise ValueError(f"{_CORES_ENV} must be in [1, {_MAX_CORES}]")
    return cores


def register():
    """Called by vLLM's general plugin loader in driver and worker processes."""
    library = os.getenv(_LIBRARY_ENV)
    diagnostics = os.getenv(_DIAGNOSTICS_ENV) == "1"
    if not library:
        if diagnostics:
            _LOG.warning("RGDR patch inactive pid=%s because %s is unset", os.getpid(), _LIBRARY_ENV)
        return
    mode = os.getenv(_MODE_ENV, "decode")
    if mode not in {"decode", "spec", "both"}:
        raise ValueError(f"{_MODE_ENV} must be decode, spec or both")
    cores = _configured_cores()
    for package in ("vllm", "vllm-ascend"):
        installed = version(package)
        if not _SUPPORTED_VERSION.fullmatch(installed):
            raise RuntimeError(f"Grouped RGDR patch expects {package} 0.23.0 or 0.23.0rc1, found {installed}")
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
    is_capturing = None
    if diagnostics:
        from vllm.forward_context import get_forward_context, is_forward_context_available

        def is_capturing():
            return is_forward_context_available() and getattr(get_forward_context(), "capturing", False)

    setattr(
        gdn,
        _HELPER,
        _wrap(
            original,
            gdn.l2norm_310p,
            op.default,
            mode,
            torch,
            cores=cores,
            diagnostics=diagnostics,
            is_capturing=is_capturing,
        ),
    )
    _LOG.info("310P grouped RGDR monkey patch installed (mode=%s, cores=%s, library=%s)", mode, cores, path)
    if diagnostics:
        _LOG.warning("RGDR patch active pid=%s mode=%s cores=%s", os.getpid(), mode, cores)
