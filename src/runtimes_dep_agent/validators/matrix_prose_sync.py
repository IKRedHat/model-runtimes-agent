"""
Align human-readable deployment prose with authoritative deployment_matrix.json.

The supervisor and Decision Specialist narratives are LLM-generated; deployability rows
are computed by deployability_engine. When every matrix row is deployable but prose
still says NO-GO, patch supervisor_summary.txt and deployment_info.txt so reports match.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def safe_matrix_entries(matrix: Any) -> list[dict[str, Any]]:
    if not isinstance(matrix, list):
        return []
    return [e for e in matrix if isinstance(e, dict)]


def all_matrix_deployable(matrix: list[dict[str, Any]]) -> bool:
    return bool(matrix) and all(e.get("deployable") is True for e in matrix)


def gpu_product_line(gpu_text: str) -> str:
    for ln in gpu_text.strip().splitlines():
        if "gpu product" in ln.lower():
            return ln.strip()
    return ""


def format_matrix_reasons_markdown(matrix: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for e in matrix:
        if not e.get("deployable"):
            continue
        name = e.get("model_name", "unknown")
        reason = str(e.get("reason", "")).strip()
        lines.append(f"- **{name}:** {reason}")
    return "\n".join(lines) if lines else "_No deployable models._"


def stored_prose_contradicts_matrix_go(deployment_text: str, summary_text: str) -> bool:
    """True when prose still says NO-GO / not deployable while matrix may be all GO."""
    dep = (deployment_text or "").lower()
    summ = (summary_text or "").lower()
    if "not deployable" in dep:
        return True
    if "no-go" in summ or "no go" in summ:
        return True
    if re.search(r"\bno-go\b", dep):
        return True
    return False


def patch_supervisor_deployment_section(summary_text: str) -> str:
    """Replace first **NO-GO** in ### Deployment Decision with GO + authoritative note."""
    if not summary_text.strip():
        return summary_text
    m = re.search(r"(###\s*Deployment Decision\s*\n)([\s\S]*?)(?=\n###\s|\Z)", summary_text, re.I)
    if not m:
        return summary_text
    header, body = m.group(1), m.group(2)
    low = body.lower()
    if "no-go" not in low and "not deployable" not in low:
        return summary_text
    new_body = body
    new_body = re.sub(
        r"\*\*NO-GO\*\*",
        "**GO** (authoritative: `deployment_matrix.json` — deterministic engine marks every model deployable)",
        new_body,
        count=1,
        flags=re.I,
    )
    if new_body == body:
        new_body = re.sub(
            r"(^|\n)(\*\*)?(NO-GO)(\*\*)?(\.|$)",
            r"\1**GO** (per deployment_matrix.json)\5",
            new_body,
            count=1,
            flags=re.I | re.MULTILINE,
        )
    return summary_text[: m.start(1)] + header + new_body + summary_text[m.end(2) :]


def align_prose_with_deployability_matrix(
    deployment_text: str,
    summary_text: str,
    matrix: list[dict[str, Any]],
    gpu_text: str,
) -> tuple[str, str]:
    """Rewrite stale LLM prose when the matrix + gpu_info prove all models deployable."""
    reasons_md = format_matrix_reasons_markdown(matrix)
    gpu_line = gpu_product_line(gpu_text)

    dep = deployment_text
    if dep:
        dep = re.sub(
            r"\*\*Decision:\*\*\s*\*\*Not Deployable\*\*",
            "**Decision:** **Deployable** (matches deployment_matrix.json; see deployability engine)",
            dep,
            count=1,
            flags=re.IGNORECASE,
        )

        def _reason_repl(m: re.Match[str]) -> str:
            return f"{m.group(1)}\n{reasons_md}\n{m.group(3)}"

        new_block, n_sub = re.subn(
            r"(####\s*Reason for Decision:\s*\n)(.*?)(\n\n---\s*\n\n###\s*Optimized Serving Arguments)",
            _reason_repl,
            dep,
            count=1,
            flags=re.DOTALL,
        )
        if n_sub:
            dep = new_block
        dep = re.sub(
            r"Even though the model is not deployable on the current hardware,\s*",
            "For this workload, ",
            dep,
            count=1,
            flags=re.IGNORECASE,
        )

    summ = summary_text
    if summ:
        summ = patch_supervisor_deployment_section(summ)
        summ = re.sub(
            r"The Decision Specialist returned a \*\*NO-GO\*\* verdict\.",
            "The Decision Specialist narrative initially suggested **NO-GO**; **deployability_matrix.json** "
            "(deterministic) marks all models deployable — **effective verdict: GO**.",
            summ,
            count=1,
        )
        summ = re.sub(
            r"The decision was based on a hardware-quantization incompatibility\..*?making deployment impossible\.",
            (
                f"Cluster inventory ({gpu_line or 'see Accelerator Summary'}) is compatible per engine rules. "
                "Any earlier NO-GO narrative contradicted the authoritative matrix."
            ),
            summ,
            count=1,
            flags=re.DOTALL,
        )
        summ = re.sub(
            r"These arguments were not applied because the deployment is not viable on the current hardware\.",
            "Apply optimized serving arguments from the Decision Specialist output when merging the model-car.",
            summ,
            count=1,
        )
        summ = re.sub(
            r"QA validation was not performed due to the \*\*NO-GO\*\* deployment decision from the Decision Specialist\.",
            "QA validation was not run in this session. With matrix **GO**, run the QA pipeline when ready.",
            summ,
            count=1,
        )

    return dep, summ


def sync_info_dir_prose_with_matrix(info_dir: Path) -> bool:
    """
    If deployment_matrix.json has every model deployable, patch supervisor_summary.txt and
    deployment_info.txt when they still say NO-GO. Returns True if any file was updated.
    """
    matrix_path = info_dir / "deployment_matrix.json"
    gpu_path = info_dir / "gpu_info.txt"
    dep_path = info_dir / "deployment_info.txt"
    summ_path = info_dir / "supervisor_summary.txt"

    try:
        raw = matrix_path.read_text(encoding="utf-8").strip()
        if not raw:
            return False
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError):
        return False

    if isinstance(data, dict):
        matrix = [data]
    elif isinstance(data, list):
        matrix = safe_matrix_entries(data)
    else:
        return False

    if not all_matrix_deployable(matrix):
        return False

    gpu_text = ""
    try:
        if gpu_path.exists():
            gpu_text = gpu_path.read_text(encoding="utf-8")
    except OSError:
        pass

    dep = ""
    summ = ""
    try:
        if dep_path.exists():
            dep = dep_path.read_text(encoding="utf-8")
    except OSError:
        pass
    try:
        if summ_path.exists():
            summ = summ_path.read_text(encoding="utf-8")
    except OSError:
        pass

    if not stored_prose_contradicts_matrix_go(dep, summ):
        return False

    dep2, summ2 = align_prose_with_deployability_matrix(dep, summ, matrix, gpu_text)
    changed = False
    if dep2 != dep:
        try:
            dep_path.write_text(dep2, encoding="utf-8")
            changed = True
        except OSError:
            pass
    if summ2 != summ:
        try:
            summ_path.write_text(summ2, encoding="utf-8")
            changed = True
        except OSError:
            pass
    return changed


__all__ = [
    "align_prose_with_deployability_matrix",
    "all_matrix_deployable",
    "safe_matrix_entries",
    "stored_prose_contradicts_matrix_go",
    "sync_info_dir_prose_with_matrix",
]
