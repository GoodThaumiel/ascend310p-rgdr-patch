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


def make_args(tokens, *, strided=False, key_heads=8):
    value_heads = 2 * key_heads
    state = torch.zeros(4, value_heads * 128 * 128 + 32, dtype=torch.float16)
    state = state[:, 16:-16].view(4, value_heads, 128, 128)
    return {
        "q": torch.zeros(1, tokens, key_heads, 128),
        "k": torch.zeros(1, tokens, key_heads, 128),
        "v": torch.zeros(1, tokens, value_heads, 128),
        "g": torch.zeros(1, tokens, value_heads),
        "beta": torch.zeros(1, tokens, value_heads),
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

    def test_decode_accepts_unsharded_rank_heads_with_padding(self):
        args = make_args(32, strided=True, key_heads=16)
        args["cu_seqlens"] = torch.tensor([0, 1, 2] + [2] * 30, dtype=torch.int32)
        args["ssm_state_indices"] = torch.tensor([3, 1] + [0] * 30, dtype=torch.int32)
        wrapped = plugin._wrap(self.original, self.normalize, self.grouped, "decode", torch)

        result = wrapped(**args)

        self.original.assert_not_called()
        q, k, v, beta, gate, state, _, _, out, *_ = self.grouped.call_args.args
        self.assertEqual(q.shape, k.shape)
        self.assertEqual(q.shape, (32, 16, 128))
        self.assertEqual(v.shape, out.shape)
        self.assertEqual(v.shape, (32, 32, 128))
        self.assertEqual(gate.shape, beta.shape)
        self.assertEqual(gate.shape, (32, 32))
        self.assertIs(state, args["state"])
        self.assertEqual(result.shape, args["v"].shape)

    def test_unsupported_state_uses_native(self):
        args = make_args(1)
        args["state"] = args["state"].float()
        args["cu_seqlens"] = torch.tensor([0, 1])
        args["ssm_state_indices"] = torch.tensor([1])
        result = plugin._wrap(self.original, self.normalize, self.grouped, "both", torch)(**args)
        self.assertIs(result, self.original.return_value)
        self.normalize.assert_not_called()
        self.grouped.assert_not_called()

    def test_diagnostics_identify_mismatched_service_geometry_once(self):
        args = make_args(1)
        args["q"] = args["q"][:, :, :4]
        args["k"] = args["k"][:, :, :4]
        args["cu_seqlens"] = torch.tensor([0, 1])
        args["ssm_state_indices"] = torch.tensor([1])
        is_capturing = Mock(side_effect=[False, True, True])
        wrapped = plugin._wrap(
            self.original, self.normalize, self.grouped, "decode", torch, diagnostics=True, is_capturing=is_capturing
        )
        with self.assertLogs(plugin.__name__, level="WARNING") as logs:
            wrapped(**args)
            wrapped(**args)
            wrapped(**args)
        self.assertEqual(len(logs.output), 2)
        self.assertIn("route=native reason=unsupported Q/K/V or gate/beta shape", logs.output[0])
        self.assertIn("q=((1, 1, 4, 128)", logs.output[0])
        self.assertIn("capturing=False", logs.output[0])
        self.assertIn("capturing=True", logs.output[1])
        self.assertEqual(self.original.call_count, 3)
        self.grouped.assert_not_called()


class TestRegistration(unittest.TestCase):
    def test_disabled_registration_does_not_import_ascend(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(plugin, "import_module") as load:
            plugin.register()
            load.assert_not_called()

    def test_loaded_once_and_only_replaces_helper(self):
        for ascend_version in ("0.23.0+local", "0.23.0rc1", "0.23.0rc1+local"):
            with self.subTest(ascend_version=ascend_version):
                self._check_registration(ascend_version)

    def _check_registration(self, ascend_version):
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
            patch.object(plugin, "version", side_effect={"vllm": "0.23.0", "vllm-ascend": ascend_version}.__getitem__),
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

    def test_other_releases_are_rejected(self):
        for ascend_version in ("0.23.0rc10", "0.23.0rc2", "0.23.1"):
            with (
                self.subTest(ascend_version=ascend_version),
                patch.dict(os.environ, {plugin._LIBRARY_ENV: "/unused/grouped.so"}),
                patch.object(plugin, "version", side_effect=["0.23.0", ascend_version]),
                patch.object(plugin, "import_module") as load,
            ):
                with self.assertRaisesRegex(RuntimeError, "expects vllm-ascend"):
                    plugin.register()
                load.assert_not_called()
