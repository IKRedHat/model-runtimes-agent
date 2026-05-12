"""Report narrative alignment when matrix + gpu_info contradict stale prose."""

from __future__ import annotations

import unittest
from pathlib import Path
import tempfile

from runtimes_dep_agent.report import html_report as hr
from runtimes_dep_agent.validators.matrix_prose_sync import align_prose_with_deployability_matrix


SAMPLE_DEPLOYMENT = """### Deployment Decision Report

**Model:** `nvidia-nemotron`

**Decision:** **Not Deployable**

#### Reason for Decision:

The primary issue is a hardware-quantization incompatibility.

---

### Optimized Serving Arguments

Even though the model is not deployable on the current hardware, the following serving arguments are recommended.
"""

SAMPLE_SUPERVISOR = """### Deployment Decision
The Decision Specialist returned a **NO-GO** verdict.

The decision was based on a hardware-quantization incompatibility. The model is FP8. The available cluster hardware (NVIDIA A100, Ampere generation) does not support FP8, making deployment impossible.

These arguments were not applied because the deployment is not viable on the current hardware.

### QA Validation
QA validation was not performed due to the **NO-GO** deployment decision from the Decision Specialist.
"""

GPU_SNIPPET = """• GPU Product: NVIDIA-H100-80GB-HBM3
• Allocatable GPUs: 8
"""


class TestMdToHtmlHeadings(unittest.TestCase):
    def test_four_hash_heading_renders_as_h5(self) -> None:
        from runtimes_dep_agent.report.html_report import _md_to_html

        html = _md_to_html("#### Reason for Decision:\n\nSome text.")
        self.assertIn('<h5 class="md-h4">Reason for Decision:</h5>', html)
        self.assertNotIn("<p>#### Reason", html)


class TestHtmlReportNarrative(unittest.TestCase):
    def test_align_narrative_rewrites_stale_no_go(self) -> None:
        matrix = [
            {
                "model_name": "nvidia-nemotron",
                "deployable": True,
                "reason": "Reconciled: H100 present.",
            }
        ]
        d, s = align_prose_with_deployability_matrix(
            SAMPLE_DEPLOYMENT, SAMPLE_SUPERVISOR, matrix, GPU_SNIPPET
        )
        self.assertIn("**Deployable**", d)
        self.assertIn("Reconciled: H100 present.", d)
        self.assertIn("deployment_matrix.json", s)
        self.assertRegex(s, r"\*\*GO\*\*")
        self.assertIn("NVIDIA-H100", s)
        self.assertNotIn("making deployment impossible", s)
        self.assertIn("run the QA pipeline", s)

    def test_generate_report_badge_go_when_matrix_all_deployable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "models_info.json").write_text("{}", encoding="utf-8")
            (base / "deployment_matrix.json").write_text(
                '[{"model_name": "m", "deployable": true, "reason": "ok"}]',
                encoding="utf-8",
            )
            (base / "gpu_info.txt").write_text(GPU_SNIPPET, encoding="utf-8")
            (base / "deployment_info.txt").write_text(SAMPLE_DEPLOYMENT, encoding="utf-8")
            (base / "supervisor_summary.txt").write_text(SAMPLE_SUPERVISOR, encoding="utf-8")
            out = base / "out.html"
            hr.generate_html_report(base, out)
            html = out.read_text(encoding="utf-8")
            self.assertIn('class="badge badge-go"', html)
            self.assertIn("callout-reconcile", html)
            self.assertNotIn('class="badge badge-nogo"', html)


if __name__ == "__main__":
    unittest.main()
