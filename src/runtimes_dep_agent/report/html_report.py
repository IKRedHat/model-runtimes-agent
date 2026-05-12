"""Generate a self-contained HTML report from the agent's info/ output files."""

from __future__ import annotations

import base64
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..validators.matrix_prose_sync import (
    align_prose_with_deployability_matrix,
    all_matrix_deployable,
    safe_matrix_entries,
    stored_prose_contradicts_matrix_go,
)

_LOGO_PATH = Path(__file__).parent / "openshift_ai_logo.png"


def generate_html_report(
    info_dir: Path,
    output_path: Path = Path("report.html"),
    agent_output: str | None = None,
    preflight_results: list[dict] | None = None,
) -> Path:
    """Read info/ artefacts and write a single self-contained HTML report."""
    models = _load_json(info_dir / "models_info.json", default={})
    matrix = _load_json(info_dir / "deployment_matrix.json", default=[])
    if isinstance(matrix, dict):
        matrix = [matrix]
    gpu_text = _load_text(info_dir / "gpu_info.txt")
    deployment_text = _load_text(info_dir / "deployment_info.txt")
    summary_text = agent_output or _load_text(info_dir / "supervisor_summary.txt")

    safe_m = safe_matrix_entries(matrix)
    all_deployable = all_matrix_deployable(safe_m)
    narrative_note = ""
    if all_deployable and stored_prose_contradicts_matrix_go(deployment_text, summary_text):
        deployment_text, summary_text = align_prose_with_deployability_matrix(
            deployment_text, summary_text, safe_m, gpu_text
        )
        narrative_note = (
            "Supervisor and deployment decision text below was aligned with the "
            "deployability matrix (deployment_matrix.json) and gpu_info.txt."
        )

    verdict = "GO" if all_deployable else _extract_verdict(deployment_text)
    gpu_nodes = _parse_gpu_nodes(gpu_text)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    logo_b64 = _load_logo_b64()

    sections = [
        _section_preflight(preflight_results),
        _section_configuration(models),
        _section_accelerator(gpu_nodes),
        _section_deployment(verdict, deployment_text, matrix, narrative_note),
        _section_qa(summary_text),
        _section_full_output(summary_text),
    ]

    body = "\n".join(s for s in sections if s)

    page = _TEMPLATE.replace("{{TIMESTAMP}}", timestamp)
    page = page.replace("{{VERDICT_BADGE}}", _verdict_badge(verdict))
    page = page.replace("{{LOGO_B64}}", logo_b64)
    page = page.replace("{{BODY}}", body)

    output_path = Path(output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(page, encoding="utf-8")
    return output_path


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _load_logo_b64() -> str:
    try:
        data = _LOGO_PATH.read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception:
        return ""


def _load_json(path: Path, default: Any = None) -> Any:
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return default
        return json.loads(text)
    except Exception:
        return default


def _load_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _machine_verdict_line_html(verdict: str) -> str:
    """Single parseable line for HTML reports (matches text markers used by _extract_verdict)."""
    v = (verdict or "UNKNOWN").strip().upper()
    if v == "NO-GO":
        label = "NO-GO"
    elif v == "GO":
        label = "GO"
    else:
        label = "UNKNOWN"
    return f'<p class="machine-verdict"><strong>Verdict: {_esc(label)}</strong></p>'


def _extract_verdict(deployment_text: str) -> str:
    """Extract verdict from explicit 'Verdict:' or 'Deployment Decision:' markers only."""
    if not deployment_text or not deployment_text.strip():
        return "UNKNOWN"
    text_lower = deployment_text.lower()
    for marker in ("verdict:", "deployment decision:"):
        idx = text_lower.find(marker)
        if idx == -1:
            continue
        rest = deployment_text[idx + len(marker) :].split("\n")[0].strip()
        tokens = rest.upper().split()
        if not tokens:
            continue
        first = tokens[0]
        if first == "GO":
            return "GO"
        if first == "NO-GO":
            return "NO-GO"
        if first == "NO" and len(tokens) >= 2 and tokens[1] == "GO":
            return "NO-GO"
    return "UNKNOWN"


def _parse_gpu_nodes(text: str) -> list[dict]:
    if not text:
        return []
    nodes: list[dict] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        node: dict[str, str] = {}
        for line in block.splitlines():
            line = line.strip().lstrip("\u2022").strip()
            if ":" in line:
                key, _, val = line.partition(":")
                node[key.strip()] = val.strip()
        if node:
            nodes.append(node)
    return nodes


# ---------------------------------------------------------------------------
# Markdown to HTML converter (lightweight, no external deps)
# ---------------------------------------------------------------------------

def _md_to_html(text: str) -> str:
    """Convert markdown text to styled HTML."""
    if not text:
        return ""
    result_lines: list[str] = []
    in_code_block = False
    code_lang = ""
    code_lines: list[str] = []
    in_list = False
    in_table = False
    table_lines: list[str] = []

    for line in text.split("\n"):
        # Code block fences
        if line.strip().startswith("```"):
            if in_code_block:
                code_content = _esc("\n".join(code_lines))
                result_lines.append(f'<pre class="code-block"><code>{code_content}</code></pre>')
                code_lines = []
                in_code_block = False
            else:
                if in_list:
                    result_lines.append("</ul>")
                    in_list = False
                in_code_block = True
                code_lang = line.strip().lstrip("`").strip()
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Table rows
        if "|" in line and line.strip().startswith("|"):
            if not in_table:
                if in_list:
                    result_lines.append("</ul>")
                    in_list = False
                in_table = True
                table_lines = []
            table_lines.append(line)
            continue
        elif in_table:
            result_lines.append(_render_table(table_lines))
            in_table = False
            table_lines = []

        stripped = line.strip()

        if not stripped:
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append("")
            continue

        # Headings (check #### before ### so four-hash lines are not misparsed)
        if stripped.startswith("#### "):
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(f'<h5 class="md-h4">{_inline_md(stripped[5:])}</h5>')
            continue
        if stripped.startswith("### "):
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(f'<h4 class="md-h3">{_inline_md(stripped[4:])}</h4>')
            continue
        if stripped.startswith("## "):
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(f'<h3 class="md-h2">{_inline_md(stripped[3:])}</h3>')
            continue
        if stripped.startswith("# "):
            if in_list:
                result_lines.append("</ul>")
                in_list = False
            result_lines.append(f'<h2 class="md-h1">{_inline_md(stripped[2:])}</h2>')
            continue

        # List items
        if re.match(r"^[-*]\s+", stripped):
            if not in_list:
                in_list = True
                result_lines.append('<ul class="md-list">')
            item_text = re.sub(r"^[-*]\s+", "", stripped)
            result_lines.append(f"  <li>{_inline_md(item_text)}</li>")
            continue
        if re.match(r"^\d+\.\s+", stripped):
            if not in_list:
                in_list = True
                result_lines.append('<ul class="md-list">')
            item_text = re.sub(r"^\d+\.\s+", "", stripped)
            result_lines.append(f"  <li>{_inline_md(item_text)}</li>")
            continue

        # Regular paragraph
        if in_list:
            result_lines.append("</ul>")
            in_list = False
        result_lines.append(f"<p>{_inline_md(stripped)}</p>")

    if in_list:
        result_lines.append("</ul>")
    if in_code_block:
        code_content = _esc("\n".join(code_lines))
        result_lines.append(f'<pre class="code-block"><code>{code_content}</code></pre>')
    if in_table:
        result_lines.append(_render_table(table_lines))

    return "\n".join(result_lines)


def _inline_md(text: str) -> str:
    """Convert inline markdown (bold, italic, code, links)."""
    text = _esc(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"__(.+?)__", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r'<code class="inline">\1</code>', text)
    return text


def _render_table(lines: list[str]) -> str:
    """Render markdown table lines as an HTML table."""
    rows = []
    for line in lines:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.append(cells)

    if len(rows) < 2:
        return ""
    # Skip separator row (row with :--- etc)
    header = rows[0]
    data_rows = [r for r in rows[1:] if not all(re.match(r"^[-:]+$", c) for c in r)]

    html_parts = ['<div class="table-wrap"><table>', "<thead><tr>"]
    for cell in header:
        html_parts.append(f"<th>{_inline_md(cell)}</th>")
    html_parts.append("</tr></thead><tbody>")
    for row in data_rows:
        html_parts.append("<tr>")
        for cell in row:
            html_parts.append(f"<td>{_inline_md(cell)}</td>")
        html_parts.append("</tr>")
    html_parts.append("</tbody></table></div>")
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------

def _verdict_badge(verdict: str) -> str:
    if verdict == "GO":
        return '<span class="badge badge-go">GO</span>'
    if verdict == "NO-GO":
        return '<span class="badge badge-nogo">NO-GO</span>'
    return '<span class="badge badge-unknown">UNKNOWN</span>'


def _esc(text: str) -> str:
    return html.escape(str(text))


def _section_preflight(results: list[dict] | None) -> str:
    if not results:
        return ""
    rows = ""
    for r in results:
        name = r.get("name", "")
        podman_engine_down = (
            name == "podman"
            and r.get("installed")
            and r.get("running") is False
        )
        if r.get("installed") and not podman_engine_down:
            engine = ""
            if name == "podman" and r.get("running") is True:
                engine = " <span class=\"status-pass\">Engine OK</span>"
            icon = f'<span class="status-pass">&#10003; Installed</span>{engine}'
        elif podman_engine_down:
            detail = _esc(r.get("running_detail") or "")
            icon = (
                '<span class="status-pass">&#10003; Installed</span> '
                '<span class="status-fail">&#10007; Engine not reachable</span>'
            )
            if detail:
                icon += f' <span class="muted">({detail})</span>'
        else:
            icon = '<span class="status-fail">&#10007; Not Found</span>'
        rows += f"<tr><td><code>{_esc(name)}</code></td><td>{icon}</td><td>{_esc(r.get('version', ''))}</td><td><code>{_esc(r.get('path', ''))}</code></td></tr>\n"
    return f"""
    <section id="preflight">
      <h2>Pre-flight Checks</h2>
      <table>
        <thead><tr><th>Tool</th><th>Status</th><th>Version</th><th>Path</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
    </section>"""


def _section_configuration(models: dict) -> str:
    if not models:
        return '<section id="configuration"><h2>Configuration Summary</h2><p class="muted">No model information available.</p></section>'
    rows = ""
    for name, m in models.items():
        p_billion = m.get("model_p_billion")
        if p_billion is not None:
            params = f"{p_billion}B" if p_billion >= 1 else f"{p_billion * 1000:.0f}M"
        else:
            params = "N/A"
        quant = f"{m.get('quantization_bits')} bits" if m.get("quantization_bits") else "N/A"
        rows += (
            f"<tr>"
            f"<td><strong>{_esc(m.get('model_name', name))}</strong></td>"
            f"<td><code>{_esc(m.get('image', ''))}</code></td>"
            f"<td>{m.get('model_size_gb', 0):.2f} GB</td>"
            f"<td>{params}</td>"
            f"<td>{quant}</td>"
            f"<td>{m.get('required_vram_gb', 'N/A')} GB</td>"
            f"<td>{_esc(m.get('supported_arch', 'N/A'))}</td>"
            f"</tr>\n"
        )
    return f"""
    <section id="configuration">
      <h2>Configuration Summary</h2>
      <p style="margin-bottom:16px"><strong>{len(models)}</strong> model(s) found in configuration.</p>
      <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Image</th><th>Size</th><th>Parameters</th><th>Quantization</th><th>Est. VRAM</th><th>Arch</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      </div>
    </section>"""


def _section_accelerator(gpu_nodes: list[dict]) -> str:
    if not gpu_nodes:
        return '<section id="accelerator"><h2>Accelerator Summary</h2><p class="muted">No GPU information available.</p></section>'
    cards = ""
    total_gpus = 0
    for i, node in enumerate(gpu_nodes, 1):
        gpus = node.get("Allocatable GPUs", "0")
        try:
            total_gpus += int(gpus)
        except ValueError:
            pass
        cards += f"""
        <div class="card">
          <h3>Node {i}: {_esc(node.get('Node Name', 'Unknown'))}</h3>
          <ul>
            <li><strong>Cloud Provider:</strong> {_esc(node.get('Cloud Provider', 'N/A'))}</li>
            <li><strong>Instance Type:</strong> {_esc(node.get('Instance Type', 'N/A'))}</li>
            <li><strong>GPU Provider:</strong> {_esc(node.get('GPU Provider', 'N/A'))}</li>
            <li><strong>GPU Product:</strong> {_esc(node.get('GPU Product', 'N/A'))}</li>
            <li><strong>Per-GPU Memory:</strong> {_esc(node.get('Per-GPU Memory', 'N/A'))}</li>
            <li><strong>Allocatable GPUs:</strong> {_esc(gpus)}</li>
            <li><strong>Node RAM:</strong> {_esc(node.get('Node RAM', 'N/A'))}</li>
            <li><strong>Node Storage:</strong> {_esc(node.get('Node Storage', 'N/A'))}</li>
          </ul>
        </div>"""
    return f"""
    <section id="accelerator">
      <h2>Accelerator Summary</h2>
      <p style="margin-bottom:16px"><strong>{len(gpu_nodes)}</strong> GPU node(s) &mdash; <strong>{total_gpus}</strong> total GPUs.</p>
      <div class="cards">{cards}</div>
    </section>"""


def _section_deployment(
    verdict: str, deployment_text: str, matrix: list, narrative_note: str = ""
) -> str:
    badge = _verdict_badge(verdict)

    note_html = ""
    if narrative_note.strip():
        note_html = f'<p class="callout-reconcile">{_esc(narrative_note.strip())}</p>'

    matrix_html = ""
    safe_matrix = [e for e in (matrix if isinstance(matrix, list) else []) if isinstance(e, dict)]
    models_map = models if isinstance(models, dict) else {}
    if safe_matrix:
        deployable = [e for e in safe_matrix if _matrix_entry_fully_deployable(e, models_map)]
        blocked = [e for e in safe_matrix if not _matrix_entry_fully_deployable(e, models_map)]
        matrix_html += '<div class="matrix-grid">'
        matrix_html += '<div class="matrix-col"><h4>Deployable</h4>'
        if deployable:
            for e in deployable:
                matrix_html += f'<div class="matrix-item pass">&#10003; <strong>{_esc(e.get("model_name", "Unknown"))}</strong><br><small>{_esc(e.get("reason", ""))}</small></div>'
        else:
            matrix_html += '<p class="muted">None</p>'
        matrix_html += "</div>"
        matrix_html += '<div class="matrix-col"><h4>Non-deployable</h4>'
        if blocked:
            for e in blocked:
                matrix_html += f'<div class="matrix-item fail">&#10007; <strong>{_esc(e.get("model_name", "Unknown"))}</strong><br><small>{_esc(e.get("reason", ""))}</small></div>'
        else:
            matrix_html += '<p class="muted">None</p>'
        matrix_html += "</div></div>"

    detail_html = ""
    machine = _machine_verdict_line_html(verdict)
    if deployment_text:
        rendered = _md_to_html(deployment_text)
        detail_html = f"""
        <div class="rendered-md" style="margin-top:16px">{machine}{rendered}</div>"""
    else:
        detail_html = f"""
        <div class="rendered-md" style="margin-top:16px">{machine}</div>"""

    return f"""
    <section id="deployment">
      <h2>Deployment Decision {badge}</h2>
      {note_html}
      {matrix_html}
      {detail_html}
    </section>"""


def _section_qa(summary_text: str) -> str:
    status = "pending"
    message = "QA validation information not found."

    if summary_text:
        for pattern in [
            r"###\s*QA\s*Validation\s*\n(.*?)(?=\n###|\Z)",
            r"##\s*QA\s*Validation\s*\n(.*?)(?=\n##|\Z)",
        ]:
            match = re.search(pattern, summary_text, re.IGNORECASE | re.DOTALL)
            if match:
                message = match.group(1).strip()
                break

        lower = message.lower()
        if any(w in lower for w in ("pass", "success", "completed successfully")):
            status = "passed"
        elif any(w in lower for w in ("fail", "error")):
            status = "failed"
        elif any(w in lower for w in ("skip", "not run", "no-go")):
            status = "skipped"

    rendered_msg = _md_to_html(message)
    status_cls = {"passed": "status-pass", "failed": "status-fail", "skipped": "status-warn"}.get(status, "status-pending")
    return f"""
    <section id="qa">
      <h2>QA Validation</h2>
      <p>Status: <span class="{status_cls}"><strong>{_esc(status.upper())}</strong></span></p>
      <div class="qa-message">{rendered_msg}</div>
    </section>"""


def _section_full_output(text: str) -> str:
    if not text:
        return ""
    rendered = _md_to_html(text)
    return f"""
    <section id="output">
      <h2>Full Agent Output</h2>
      <div class="rendered-md">{rendered}</div>
    </section>"""


# ---------------------------------------------------------------------------
# Self-contained HTML template
# ---------------------------------------------------------------------------

_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Model Runtimes Agent &mdash; OpenShift AI</title>
<style>
:root {
  --bg: #f5f6fa;
  --surface: #ffffff;
  --sidebar-bg: #1e2a38;
  --sidebar-fg: #c8d6e5;
  --text: #2d3436;
  --text-muted: #636e72;
  --border: #dfe6e9;
  --accent: #0984e3;
  --green: #00b894;
  --red: #d63031;
  --yellow: #fdcb6e;
  --radius: 8px;
}
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, sans-serif;
  background: var(--bg);
  color: var(--text);
  display: flex;
  min-height: 100vh;
}

/* Sidebar */
.sidebar {
  width: 250px;
  background: #ffffff;
  color: var(--text);
  padding: 20px 16px;
  flex-shrink: 0;
  position: fixed;
  top: 0; bottom: 0;
  overflow-y: auto;
  border-right: 1px solid var(--border);
}
.sidebar .logo { display: block; width: 180px; margin: 0 auto 20px auto; }
.sidebar nav a {
  display: block;
  padding: 10px 14px;
  color: #ffffff;
  background: var(--sidebar-bg);
  text-decoration: none;
  border-radius: var(--radius);
  margin-bottom: 4px;
  font-size: 0.85rem;
  font-weight: 500;
  transition: background 0.15s, transform 0.1s;
}
.sidebar nav a:hover { background: #2c3e50; transform: translateX(2px); }

/* Main */
.main {
  margin-left: 250px;
  flex: 1;
  padding: 32px 48px 32px 48px;
  min-width: 0;
}
.header {
  display: flex;
  align-items: center;
  gap: 16px;
  margin-bottom: 32px;
  flex-wrap: wrap;
  background: var(--sidebar-bg);
  color: #fff;
  padding: 20px 28px;
  border-radius: var(--radius);
}
.header h1 { font-size: 1.5rem; color: #fff; }
.badge {
  display: inline-block;
  padding: 5px 16px;
  border-radius: 20px;
  font-weight: 700;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}
.badge-go { background: var(--green); color: #fff; }
.badge-nogo { background: var(--red); color: #fff; }
.badge-unknown { background: var(--yellow); color: #333; }

.callout-reconcile {
  background: #e8f4fd;
  border-left: 4px solid var(--accent);
  padding: 12px 16px;
  margin-bottom: 16px;
  border-radius: var(--radius);
  font-size: 0.9rem;
}

/* Sections */
section {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 28px 32px;
  margin-bottom: 20px;
}
section h2 { font-size: 1.15rem; margin-bottom: 14px; color: var(--sidebar-bg); border-bottom: 2px solid var(--border); padding-bottom: 8px; }

/* Tables */
table { width: 100%; border-collapse: collapse; font-size: 0.85rem; }
th, td { text-align: left; padding: 10px 14px; border-bottom: 1px solid var(--border); }
th { background: var(--bg); font-weight: 600; color: var(--sidebar-bg); }
tr:hover { background: #f8f9fc; }
code, code.inline { background: #eef1f5; padding: 2px 6px; border-radius: 4px; font-size: 0.82rem; color: #d63031; }
.table-wrap { overflow-x: auto; }
.muted { color: var(--text-muted); font-style: italic; }

/* Cards */
.cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px; }
.card {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 18px;
}
.card h3 { font-size: 0.9rem; margin-bottom: 10px; color: var(--accent); }
.card ul { list-style: none; font-size: 0.83rem; }
.card li { margin-bottom: 5px; }

/* Matrix */
.matrix-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 14px; }
.matrix-col h4 { margin-bottom: 8px; color: var(--sidebar-bg); }
.matrix-item { padding: 12px 16px; border-radius: var(--radius); margin-bottom: 8px; font-size: 0.85rem; }
.matrix-item.pass { background: #e6fcf5; border-left: 4px solid var(--green); }
.matrix-item.fail { background: #ffeaea; border-left: 4px solid var(--red); }
.matrix-item small { color: var(--text-muted); line-height: 1.5; }

/* Status */
.status-pass { color: var(--green); }
.status-fail { color: var(--red); }
.status-warn { color: #e17055; }
.status-pending { color: var(--text-muted); }

/* QA message */
.qa-message {
  background: var(--bg);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 20px;
  margin-top: 10px;
  font-size: 0.875rem;
  line-height: 1.7;
}

/* Rendered markdown */
.rendered-md {
  padding: 16px 0;
  font-size: 0.875rem;
  line-height: 1.7;
}
.rendered-md h2.md-h1 { font-size: 1.2rem; margin: 20px 0 8px; color: var(--sidebar-bg); border-bottom: 2px solid var(--border); padding-bottom: 6px; }
.rendered-md h3.md-h2 { font-size: 1.05rem; margin: 18px 0 6px; color: var(--sidebar-bg); }
.rendered-md h4.md-h3 { font-size: 0.95rem; margin: 14px 0 6px; color: var(--accent); }
.rendered-md h5.md-h4 { font-size: 0.88rem; margin: 12px 0 4px; color: var(--text); font-weight: 600; }
.rendered-md p { margin: 6px 0; }
.rendered-md p.machine-verdict {
  font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
  font-size: 0.88rem;
  margin: 0 0 14px 0;
  padding: 10px 14px;
  background: #eef2f6;
  border-left: 4px solid var(--accent);
  border-radius: 4px;
  color: var(--text);
}
.rendered-md ul.md-list { margin: 6px 0 6px 20px; }
.rendered-md ul.md-list li { margin-bottom: 4px; }
.rendered-md .code-block {
  background: #1e272e;
  color: #a8e6cf;
  padding: 16px 20px;
  border-radius: var(--radius);
  font-size: 0.8rem;
  line-height: 1.6;
  overflow-x: auto;
  margin: 10px 0;
}
.rendered-md .code-block code { background: none; color: inherit; padding: 0; font-size: inherit; }
.rendered-md table { margin: 10px 0; }

/* Details / Expand */
details { margin-top: 10px; }
details summary {
  cursor: pointer;
  font-weight: 600;
  color: var(--accent);
  font-size: 0.875rem;
  padding: 6px 0;
}

.footer {
  text-align: center;
  color: var(--text-muted);
  font-size: 0.75rem;
  padding: 24px 0;
}
@media (max-width: 800px) {
  .sidebar { display: none; }
  .main { margin-left: 0; padding: 16px; }
  .matrix-grid { grid-template-columns: 1fr; }
}
</style>
</head>
<body>

<aside class="sidebar">
  <img src="data:image/png;base64,{{LOGO_B64}}" alt="OpenShift AI" class="logo">
  <nav>
    <a href="#preflight">Pre-flight Checks</a>
    <a href="#configuration">Configuration</a>
    <a href="#accelerator">Accelerator</a>
    <a href="#deployment">Deployment Decision</a>
    <a href="#qa">QA Validation</a>
    <a href="#output">Full Output</a>
  </nav>
</aside>

<div class="main">
  <div class="header">
    <h1>Model Runtimes Agent &mdash; OpenShift AI</h1>
  </div>

  {{BODY}}

  <div class="footer">
    Model Runtimes Agent &bull; Red Hat OpenShift AI &bull; Developed by <a href="https://github.com/Raghul-M" target="_blank" style="color: var(--accent); text-decoration: none;">Raghul M</a>
  </div>
</div>

</body>
</html>
"""
