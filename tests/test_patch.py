# SPDX-License-Identifier: Apache-2.0
import ast
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import ascend310p_rgdr_patch as plugin
import torch


def make_args(tokens, *, strided=False):
    state = torch.zeros(4, 16 * 128 * 128 + 32, dtype=torch.float16)
    state = state[:, 16:-16].view(4, 16, 128, 128)
    return {
        "q": torch.zeros(1, tokens, 8, 128),
        "k": torch.zeros(1, tokens, 8, 128),
        "v": torch.zeros(1, tokens, 16, 128),
        "g": torch.zeros(1, tokens, 16),
        "beta": torch.zeros(1, tokens, 16),
        "state": state if strided else state.contiguous(),
    }


class TestDispatch(unittest.TestCase):
    def setUp(self):
        self.original = Mock()
        self.normalize = Mock(side_effect=lambda x: x)
        self.grouped = Mock()

    def test_plain_decode_preserves_slots_and_page_stride(self):
        args = make_args(3, strided=True)
        args["cu_seqlens"] = torch.tensor([0, 1, 2, 2])
        args["ssm_state_indices"] = torch.tensor([3, 1, 0])
        result = plugin._wrap(self.original, self.normalize, self.grouped, "decode", torch)(**args)
        q, k, v, beta, g, state, slots, accepted, out, scale, deq, cores, grouped, offsets = self.grouped.call_args.args
        self.original.assert_not_called()
        self.assertEqual(self.normalize.call_count, 2)
        self.assertIs(state, args["state"])
        self.assertFalse(state.is_contiguous())
        self.assertEqual(result.shape, args["v"].shape)
        self.assertEqual(result.data_ptr(), out.data_ptr())
        self.assertTrue(all(t.dtype == torch.float16 for t in (q, k, v, beta, g)))
        torch.testing.assert_close(slots, torch.tensor([[3], [1], [0]], dtype=torch.int32))
        torch.testing.assert_close(accepted, torch.ones(3, dtype=torch.int32))
        torch.testing.assert_close(offsets, args["cu_seqlens"].int())
        self.assertEqual((scale, deq, cores, grouped), (128**-0.5, False, 7, True))

    def test_spec_mode_preserves_previous_slot_outside_short_window(self):
        args = make_args(3)
        args["cu_seqlens"] = torch.tensor([0, 1, 3])
        args["ssm_state_indices"] = torch.tensor([[1, 2], [3, 4], [0, 0]])
        args["num_accepted_tokens"] = torch.tensor([2, 1, 0])
        args["use_qk_l2norm_in_kernel"] = False
        plugin._wrap(self.original, self.normalize, self.grouped, "decode", torch)(**args)
        self.original.assert_called_once()
        self.grouped.assert_not_called()

        plugin._wrap(self.original, self.normalize, self.grouped, "spec", torch)(**args)
        self.normalize.assert_not_called()
        _, _, _, _, _, _, slots, accepted, *_ = self.grouped.call_args.args
        torch.testing.assert_close(slots, torch.tensor([[1, 2], [3, 4]], dtype=torch.int32))
        torch.testing.assert_close(accepted, torch.tensor([2, 1], dtype=torch.int32))

    def test_unsupported_state_uses_native(self):
        args = make_args(1)
        args["state"] = args["state"].float()
        args["cu_seqlens"] = torch.tensor([0, 1])
        args["ssm_state_indices"] = torch.tensor([1])
        result = plugin._wrap(self.original, self.normalize, self.grouped, "both", torch)(**args)
        self.assertIs(result, self.original.return_value)
        self.normalize.assert_not_called()
        self.grouped.assert_not_called()


class TestRegistration(unittest.TestCase):
    def test_disabled_registration_does_not_import_ascend(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(plugin, "import_module") as load:
            plugin.register()
            load.assert_not_called()

    def test_loaded_once_and_only_replaces_helper(self):
        source = f"def {plugin._HELPER}({', '.join(plugin._PARAMETERS)}): pass"
        module = SimpleNamespace(l2norm_310p=Mock())
        ns = {}
        exec(compile(ast.parse(source), "<gdn stub>", "exec"), ns)
        original = ns[plugin._HELPER]
        setattr(module, plugin._HELPER, original)
        grouped = Mock()
        grouped.default._schema.arguments = [SimpleNamespace(name="row_offsets")]
        with (
            TemporaryDirectory() as directory,
            patch.object(plugin, "import_module", return_value=module),
            patch.object(plugin, "version", return_value="0.23.0+local"),
            patch.dict("sys.modules", {"vllm_ascend.utils": SimpleNamespace(is_310p=lambda: True)}),
            patch.object(torch.ops, "load_library") as load,
            patch.object(torch.ops.rgdr_v310_cube_fp16, "run_", grouped, create=True),
        ):
            library = Path(directory) / "grouped.so"
            library.touch()
            with patch.dict(os.environ, {plugin._LIBRARY_ENV: str(library)}):
                plugin.register()
                wrapped = getattr(module, plugin._HELPER)
                plugin.register()
            load.assert_called_once_with(str(library))
        self.assertIsNot(wrapped, original)
        self.assertIs(wrapped, getattr(module, plugin._HELPER))
