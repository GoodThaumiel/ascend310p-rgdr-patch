# SPDX-License-Identifier: Apache-2.0
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from ascend310p_rgdr_patch import prefill_probe


class TestPrefillProbeTiming(unittest.TestCase):
    def test_synchronized_batches_with_partial_batch(self):
        npu = Mock()
        npu.Event.side_effect = RuntimeError("event timestamp invalid (107006)")
        operation = Mock(side_effect=range(8))

        with patch.object(
            prefill_probe.time,
            "perf_counter_ns",
            side_effect=[0, 60_000, 70_000, 130_000],
        ):
            timing, output = prefill_probe.time_op(
                SimpleNamespace(npu=npu),
                operation,
                warmup=2,
                iterations=6,
                batch_size=4,
            )

        self.assertEqual(operation.call_count, 8)
        self.assertEqual(npu.synchronize.call_count, 3)
        npu.Event.assert_not_called()
        self.assertEqual(output, 7)
        self.assertEqual(timing["sample_batch_sizes"], [4, 2])
        self.assertEqual(timing["p50_us"], 22.5)
        self.assertEqual(timing["p90_us"], 30.0)
        self.assertEqual(timing["host_wall_us_per_call"], 20.0)

    def test_device_execution_error_propagates(self):
        npu = Mock()
        npu.synchronize.side_effect = [None, RuntimeError("execution failed")]

        with self.assertRaisesRegex(RuntimeError, "execution failed"):
            prefill_probe.time_op(
                SimpleNamespace(npu=npu), Mock(), warmup=0, iterations=2, batch_size=2
            )

    def test_ratios_require_matching_timing_methods(self):
        current = {
            "name": "topk/1024x256/native",
            "status": "PASSED",
            "fixture_sha256": "same-fixture",
            "timing": {
                "p50_us": 100,
                "source": "host_synchronized_batches",
                "sample_batch_sizes": [5],
            },
        }
        reference = {**current, "timing": {"p50_us": 50}}
        row = prefill_probe.compare_reports([current], {"cases": [reference]})[0]
        self.assertFalse(row["same_timing_method"])
        self.assertNotIn("p50_ratio", row)

        reference["timing"] = {**current["timing"], "p50_us": 50}
        row = prefill_probe.compare_reports([current], {"cases": [reference]})[0]
        self.assertTrue(row["same_timing_method"])
        self.assertEqual(row["p50_ratio"], 2)
