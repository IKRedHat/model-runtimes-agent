"""Tests for deployment fit / vLLM tensor-parallel heuristics."""

from __future__ import annotations

import unittest

from runtimes_dep_agent.validators.deployability_engine import (
    min_tensor_parallel_for_weights,
    parse_gpu_inventory,
)


class TestMinTensorParallel(unittest.TestCase):
    def test_nemotron_on_80gb(self) -> None:
        # ~120 GB weights on ~80 GB GPUs → minimum TP 2
        self.assertEqual(min_tensor_parallel_for_weights(119.58, 80.0), 2)
        self.assertEqual(min_tensor_parallel_for_weights(119.58, 79.65), 2)

    def test_fits_single_gpu(self) -> None:
        self.assertEqual(min_tensor_parallel_for_weights(40.0, 80.0), 1)

    def test_edge_cases(self) -> None:
        self.assertEqual(min_tensor_parallel_for_weights(0, 80), 1)
        self.assertEqual(min_tensor_parallel_for_weights(100, 0), 1)


class TestParseGpuInventory(unittest.TestCase):
    def test_prefers_explicit_per_gpu_over_product_name(self) -> None:
        text = """• GPU Product: NVIDIA-H100-80GB-HBM3
• Per-GPU Memory: 79.65 GB GB
• Allocatable GPUs: 8
"""
        inv = parse_gpu_inventory(text)
        self.assertEqual(inv.allocatable_gpus, 8)
        self.assertEqual(inv.per_gpu_mem_gb, 79.65)
        self.assertEqual(inv.accelerator_family, "hopper")

    def test_falls_back_to_product_when_no_explicit_mem(self) -> None:
        text = """• GPU Product: NVIDIA-H100-80GB-HBM3
• Allocatable GPUs: 8
"""
        inv = parse_gpu_inventory(text)
        self.assertEqual(inv.per_gpu_mem_gb, 80.0)


if __name__ == "__main__":
    unittest.main()
