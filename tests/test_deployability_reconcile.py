"""Tests for FP8 / GPU generation reconciliation."""

from __future__ import annotations

import unittest

from runtimes_dep_agent.validators.deployability_reconcile import (
    cluster_supports_fp8_from_gpu_text,
    reconcile_deployment_matrix_entries,
)


GPU_H100 = """• GPU Product: NVIDIA-H100-80GB-HBM3
• Allocatable GPUs: 8
"""

NEMOTRON_REASON = (
    "The model is FP8 quantized, which requires NVIDIA Hopper (e.g., H100) or Ada generation GPUs. "
    "The available 80GB GPUs are likely Ampere (A100), which do not support FP8 kernels."
)


class TestDeployabilityReconcile(unittest.TestCase):
    def test_cluster_detects_h100(self) -> None:
        self.assertTrue(cluster_supports_fp8_from_gpu_text(GPU_H100))
        self.assertFalse(cluster_supports_fp8_from_gpu_text("• GPU Product: NVIDIA-A100-SXM4-80GB"))

    def test_flips_fp8_false_negative_when_h100(self) -> None:
        rows = [
            {
                "model_name": "nvidia-nemotron",
                "deployable": False,
                "reason": NEMOTRON_REASON,
            }
        ]
        out = reconcile_deployment_matrix_entries(rows, GPU_H100)
        self.assertEqual(len(out), 1)
        self.assertTrue(out[0]["deployable"])
        self.assertIn("Reconciled", out[0]["reason"])
        self.assertIn("H100", out[0]["reason"])

    def test_no_flip_when_insufficient_vram(self) -> None:
        rows = [
            {
                "model_name": "huge",
                "deployable": False,
                "reason": "FP8 model but insufficient GPU memory for weights.",
            }
        ]
        out = reconcile_deployment_matrix_entries(rows, GPU_H100)
        self.assertFalse(out[0]["deployable"])

    def test_no_flip_without_fp8_reason(self) -> None:
        rows = [{"model_name": "x", "deployable": False, "reason": "Wrong license"}]
        out = reconcile_deployment_matrix_entries(rows, GPU_H100)
        self.assertFalse(out[0]["deployable"])


if __name__ == "__main__":
    unittest.main()
