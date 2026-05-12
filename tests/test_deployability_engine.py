"""Deterministic deployability_engine tests."""

from __future__ import annotations

import unittest

from runtimes_dep_agent.validators.deployability_engine import (
    compute_deployment_matrix,
    infer_accelerator_family,
    infer_quantization_kind,
    parse_gpu_inventory,
)


GPU_H100_INFO = """• GPU Provider: NVIDIA
• GPU Product: NVIDIA-H100-80GB-HBM3
• Per-GPU Memory: 79.65 GB GB
• Allocatable GPUs: 8
"""

GPU_A100_INFO = """• GPU Provider: NVIDIA
• GPU Product: NVIDIA-A100-SXM4-80GB
• Per-GPU Memory: 79.25 GB
• Allocatable GPUs: 8
"""


class TestAcceleratorFamily(unittest.TestCase):
    def test_h100_is_hopper_not_inferred_from_gb(self) -> None:
        self.assertEqual(
            infer_accelerator_family("NVIDIA-H100-80GB-HBM3", "NVIDIA"),
            "hopper",
        )

    def test_a100_is_ampere(self) -> None:
        self.assertEqual(infer_accelerator_family("NVIDIA-A100-SXM4-80GB", "NVIDIA"), "ampere")


class TestFp8AndCapacity(unittest.TestCase):
    def test_h100_fp8_nemotron_deployable_tp2(self) -> None:
        models = {
            "nvidia-nemotron": {
                "model_name": "nvidia-nemotron",
                "image": "oci://registry.example/modelcar-nvidia-nemotron-3-fp8:1.0",
                "model_size_gb": 119.58,
                "required_vram_gb": None,
            }
        }
        rows = compute_deployment_matrix(models, GPU_H100_INFO)
        self.assertEqual(len(rows), 1)
        self.assertTrue(rows[0]["deployable"])
        r = rows[0]["reason"].lower()
        self.assertIn("hopper", r)
        self.assertIn("tensor-parallel-size=2", rows[0]["reason"])
        self.assertIn("h100", r)

    def test_a100_fp8_blocked(self) -> None:
        models = {
            "m": {
                "model_name": "m",
                "image": "oci://x/modelcar-something-fp8:y",
                "model_size_gb": 40.0,
            }
        }
        rows = compute_deployment_matrix(models, GPU_A100_INFO)
        self.assertFalse(rows[0]["deployable"])
        self.assertIn("FP8", rows[0]["reason"])
        self.assertIn("ampere", rows[0]["reason"].lower())

    def test_insufficient_gpus_for_tp(self) -> None:
        gpu_small = """• GPU Product: NVIDIA-H100-80GB-HBM3
• Per-GPU Memory: 79.65 GB
• Allocatable GPUs: 1
"""
        models = {
            "big": {
                "model_name": "big",
                "image": "oci://x/huge-model:latest",
                "model_size_gb": 119.58,
            }
        }
        rows = compute_deployment_matrix(models, gpu_small)
        self.assertFalse(rows[0]["deployable"])
        self.assertIn("allocatable", rows[0]["reason"].lower())

    def test_per_gpu_mem_used_for_tp_not_product_80(self) -> None:
        inv = parse_gpu_inventory(GPU_H100_INFO)
        self.assertEqual(inv.per_gpu_mem_gb, 79.65)

    def test_unknown_quant_proceeds_capacity_only(self) -> None:
        models = {
            "plain": {
                "model_name": "plain",
                "image": "oci://x/modelcar-plain-8b-instruct:1",
                "model_size_gb": 20.0,
            }
        }
        rows = compute_deployment_matrix(models, GPU_H100_INFO)
        self.assertTrue(rows[0]["deployable"])


class TestQuantInference(unittest.TestCase):
    def test_fp8_from_image_path(self) -> None:
        info = {"model_name": "x", "image": "oci://registry/foo/bar-fp8:3.0"}
        self.assertEqual(infer_quantization_kind("x", info), "fp8")


if __name__ == "__main__":
    unittest.main()
