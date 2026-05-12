import streamlit as st
import yaml
import os
import html
import shlex
import tempfile
import time
import subprocess
import shutil
import webbrowser
import re
import base64
from typing import List, Dict, Optional
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd
import json
from pathlib import Path
import selectors
import shutil
import uuid

from runtimes_dep_agent.utils.path_utils import detect_repo_root
from runtimes_dep_agent.preflight import run_preflight_checks, preflight_ok
from runtimes_dep_agent.report.html_report import generate_html_report

_LOGO_PATH = Path(__file__).resolve().parent / "src" / "runtimes_dep_agent" / "report" / "openshift_ai_logo.png"


def _load_logo_b64() -> str:
    try:
        return base64.b64encode(_LOGO_PATH.read_bytes()).decode("ascii")
    except Exception:
        return ""


# Page configuration
st.set_page_config(
    page_title="Model Runtimes Agent — OpenShift AI",
    page_icon=":material/smart_toy:",
    layout="wide"
)

_THEME_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Red+Hat+Display:wght@400;500;600;700&family=Red+Hat+Text:wght@400;500;600&display=swap');

:root {
    --bg: #f5f6fa;
    --surface: #ffffff;
    --sidebar-bg: #1e2a38;
    --sidebar-fg: #c8d6e5;
    --text: #2d3436;
    --text-muted: #636e72;
    --border: #dfe6e9;
    --accent: #e63946;
    --green: #00b894;
    --red: #d63031;
    --yellow: #fdcb6e;
    --radius: 8px;
}

.stApp {
    font-family: 'Red Hat Text', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

.stMainBlockContainer {
    padding-top: 0.5rem !important;
}

section[data-testid="stSidebar"] {
    background-color: #1e2a38;
    border-right: none;
}

section[data-testid="stSidebar"] .stMarkdown h1,
section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    font-family: 'Red Hat Display', sans-serif;
    color: #ffffff;
}

section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown label,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div {
    color: #ffffff;
}

section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] label p {
    color: #ffffff !important;
}

section[data-testid="stSidebar"] .stTextInput input {
    background-color: #ffffff !important;
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] .stTextInput input::placeholder {
    color: #636e72 !important;
    opacity: 1 !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    background-color: #ffffff !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] span {
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] .stButton > button {
    background-color: #ffffff !important;
    color: #1e2a38 !important;
    border: 1px solid #dfe6e9 !important;
}

section[data-testid="stSidebar"] .stButton > button * {
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] .stButton > button:hover {
    background-color: #f0f0f0 !important;
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] {
    background-color: #ffffff !important;
    border-radius: 6px;
}

section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] * {
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderDropzone"] small {
    color: #636e72 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: #2c3e50 !important;
}

section[data-testid="stSidebar"] div[data-testid="stExpander"] {
    background-color: #ffffff !important;
    border: 1px solid #dfe6e9 !important;
    border-radius: 6px;
}

section[data-testid="stSidebar"] div[data-testid="stExpander"] * {
    color: #1e2a38 !important;
}

section[data-testid="stSidebar"] div[data-testid="stExpander"] code {
    color: #1e2a38 !important;
    background-color: #f5f6fa !important;
}

section[data-testid="stSidebar"] div[data-testid="stExpander"] pre {
    background-color: #f5f6fa !important;
    color: #1e2a38 !important;
}

h1, h2, h3, h4 {
    font-family: 'Red Hat Display', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    color: var(--sidebar-bg);
}

.stProgress > div > div > div > div {
    background-color: #e63946;
}

div.stButton > button[kind="primary"] {
    background-color: #e63946;
    border: none;
    border-radius: var(--radius);
    font-weight: 600;
    font-family: 'Red Hat Display', sans-serif;
    letter-spacing: 0.3px;
    color: #fff;
}

div.stButton > button[kind="primary"]:hover {
    background-color: #c5303b;
    border: none;
    color: #fff;
}

div.stDownloadButton > button {
    background-color: var(--sidebar-bg);
    color: white;
    border: none;
    border-radius: var(--radius);
    font-weight: 600;
    font-family: 'Red Hat Display', sans-serif;
}

div.stDownloadButton > button:hover {
    background-color: #2c3e50;
    color: white;
    border: none;
}

div[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: var(--radius);
    background: var(--surface);
}

div[data-testid="stMetric"] {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
}

div[data-testid="stMetric"] label {
    color: var(--text-muted);
    font-family: 'Red Hat Display', sans-serif;
}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--sidebar-bg);
    font-family: 'Red Hat Display', sans-serif;
    font-weight: 700;
}

.stTabs [data-baseweb="tab-list"] button {
    font-family: 'Red Hat Display', sans-serif;
    font-weight: 600;
    color: var(--text-muted);
}

.stTabs [data-baseweb="tab-list"] button[aria-selected="true"] {
    color: var(--accent);
    border-bottom-color: var(--accent);
}

.stAlert {
    border-radius: var(--radius);
}

section[data-testid="stSidebar"] > div:first-child {
    padding-top: 0 !important;
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    padding-top: 0.5rem !important;
}

section[data-testid="stSidebar"] .stMarkdown h1 {
    margin-top: 0.25rem;
    margin-bottom: 0.25rem;
}

section[data-testid="stSidebar"] .stMarkdown h2,
section[data-testid="stSidebar"] .stMarkdown h3 {
    margin-top: 0.25rem;
    margin-bottom: 0;
    font-size: 0.92rem;
}

section[data-testid="stSidebar"] hr {
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
}

section[data-testid="stSidebar"] .stElementContainer {
    margin-bottom: 0 !important;
}

section[data-testid="stSidebar"] .stTextInput input {
    border: 1px solid #dfe6e9 !important;
    border-radius: 6px;
}

section[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #dfe6e9 !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stSelectbox input {
    border: none !important;
    box-shadow: none !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] {
    border: 1px solid #dfe6e9 !important;
    border-radius: 6px !important;
    outline: none !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] div[class*="control"],
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div,
section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div > div {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    background: transparent !important;
}

section[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"]:focus-within {
    border-color: #1e2a38 !important;
    box-shadow: 0 0 0 2px rgba(30, 42, 56, 0.15) !important;
}

button[data-testid="stBaseButton-secondary"][kind="secondary"]:has(+ div #stop_agent_btn),
div[data-testid="stButton"] button[kind="secondary"] {
    background-color: #e63946 !important;
    color: #ffffff !important;
    border: none !important;
    font-weight: 600;
}

div[data-testid="stButton"] button[kind="secondary"]:hover {
    background-color: #c5303b !important;
    color: #ffffff !important;
}

div[data-testid="stButton"] button[kind="secondary"] * {
    color: #ffffff !important;
}
</style>
"""
st.markdown(_THEME_CSS, unsafe_allow_html=True)

# Initialize session state
if "agent_started" not in st.session_state:
    st.session_state.agent_started = False
if "workflow_completed" not in st.session_state:
    st.session_state.workflow_completed = False
if "workflow_step" not in st.session_state:
    st.session_state.workflow_step = 0
if "start_time" not in st.session_state:
    st.session_state.start_time = None
if "yaml_config" not in st.session_state:
    st.session_state.yaml_config = None
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = None
if "oci_pull_secret" not in st.session_state:
    st.session_state.oci_pull_secret = None
if "extracted_data" not in st.session_state:
    st.session_state.extracted_data = None
if "agent_result" not in st.session_state:
    st.session_state.agent_result = None
if "agent_output_text" not in st.session_state:
    st.session_state.agent_output_text = None
if "agent_command_output" not in st.session_state:
    st.session_state.agent_command_output = None
if "agent_start_time" not in st.session_state:
    st.session_state.agent_start_time = None
if "agent_timestamps" not in st.session_state:
    st.session_state.agent_timestamps = {
        "supervisor": None,
        "configuration": None,
        "accelerator": None,
        "deployment": None,
        "qa": None
    }
if "vllm_runtime_image" not in st.session_state:
    # Check for environment variable first
    vllm_image_env = os.environ.get("VLLM_RUNTIME_IMAGE")
    st.session_state.vllm_runtime_image = vllm_image_env if vllm_image_env else None
if "runtime_backend" not in st.session_state:
    st.session_state.runtime_backend = None
if "oc_login_command" not in st.session_state:
    st.session_state.oc_login_command = None
if "preflight_results" not in st.session_state:
    st.session_state.preflight_results = None
if "agent_pid" not in st.session_state:
    st.session_state.agent_pid = None
if "registry_host" not in st.session_state:
    st.session_state.registry_host = os.environ.get("REGISTRY_HOST") or None
if "agent_interrupted" not in st.session_state:
    st.session_state.agent_interrupted = False
if "qa_skip_namespace_delete" not in st.session_state:
    st.session_state.qa_skip_namespace_delete = False
if "agent_run_info_dir" not in st.session_state:
    st.session_state.agent_run_info_dir = None


def _maybe_debug_info_symlink(root: Path, target: Path) -> None:
    """Optional: repo_root/info_debug_latest -> this run dir when AGENT_RUN_INFO_SYMLINK is truthy (local IDE/debug)."""
    raw = os.environ.get("AGENT_RUN_INFO_SYMLINK", "").strip().lower()
    if raw not in ("1", "true", "yes", "y"):
        return
    link = root / "info_debug_latest"
    try:
        if link.is_symlink() or link.is_file():
            link.unlink(missing_ok=True)
        elif link.is_dir():
            return
        link.symlink_to(target.resolve(), target_is_directory=True)
    except OSError:
        pass


def _ensure_run_info_dir() -> Path:
    """Create a new per-run info directory under <repo>/.streamlit_agent_runs/; remove the previous run dir in this session."""
    root = detect_repo_root([Path(__file__).resolve()])
    base = root / ".streamlit_agent_runs"
    base.mkdir(parents=True, exist_ok=True)
    old = st.session_state.get("agent_run_info_dir")
    if isinstance(old, Path) and old.is_dir():
        shutil.rmtree(old, ignore_errors=True)
    run_id = uuid.uuid4().hex
    d = base / run_id
    d.mkdir(parents=False, exist_ok=False)
    st.session_state["agent_run_info_dir"] = d
    _maybe_debug_info_symlink(root, d)
    return d


def _agent_subprocess_is_running() -> bool:
    """True while the first workflow step is waiting for CLI output (subprocess active)."""
    return (
        st.session_state.workflow_step == 1
        and not st.session_state.workflow_completed
        and not st.session_state.agent_command_output
    )


def _prepare_rerun_from_session() -> None:
    """
    Reset run-scoped workflow state so the supervisor runs again with the same
    sidebar inputs (API key, pull secret, YAML, oc login, registry, etc.).
    """
    old_run = st.session_state.get("run_info_dir")
    if old_run:
        p = Path(old_run)
        if p.exists():
            try:
                shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    st.session_state.run_info_dir = None
    st.session_state.workflow_completed = False
    st.session_state.workflow_step = 1
    st.session_state.agent_command_output = None
    st.session_state.agent_output_text = None
    st.session_state.agent_start_time = None
    st.session_state.agent_timestamps = {
        "supervisor": None,
        "configuration": None,
        "accelerator": None,
        "deployment": None,
        "qa": None,
    }
    st.session_state.agent_result = None
    st.session_state.agent_interrupted = False
    st.session_state.agent_pid = None
    st.session_state.start_time = time.time()
    st.session_state.pop("last_html_report_path", None)

    if st.session_state.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
    if st.session_state.oci_pull_secret:
        os.environ["OCI_REGISTRY_PULL_SECRET"] = st.session_state.oci_pull_secret

    if not st.session_state.yaml_config:
        raise ValueError("No YAML configuration in session; upload a modelcar YAML before rerunning.")

    temp_dir = tempfile.gettempdir()
    config_path = os.path.join(temp_dir, f"modelcar_config_rerun_{int(time.time() * 1000)}.yaml")
    with open(config_path, "wb") as tmp_file:
        if getattr(st.session_state, "yaml_content_raw", None):
            tmp_file.write(st.session_state.yaml_content_raw)
        else:
            yaml.dump(st.session_state.yaml_config, tmp_file, default_flow_style=False)
    st.session_state.config_path = config_path

# Helper function to get value from extracted data with fallback
def get_value(path: str, default):
    """Get nested value from extracted_data using dot notation path."""
    if not st.session_state.extracted_data:
        return default
    
    keys = path.split('.')
    value = st.session_state.extracted_data
    try:
        for key in keys:
            value = value[key]
        return value if value is not None else default
    except (KeyError, TypeError):
        return default

# Function to load model information from info/models_info.json
def load_model_info_from_json():
    """Load model information from the repo's info/models_info.json file."""
    models_info_path = _get_info_dir() / "models_info.json"
    models = []
    
    if not models_info_path.exists():
        return {"num_models": 0, "models": []}
    
    # Don't show data if agent hasn't started yet
    if st.session_state.agent_start_time is None:
        return {"num_models": 0, "models": []}
    
    # Check if file has been updated since agent started (only show if new data)
    try:
        file_mtime = models_info_path.stat().st_mtime
        # Only show if file was modified after agent started
        if file_mtime < st.session_state.agent_start_time:
            return {"num_models": 0, "models": []}
        
        # Check if file is empty (from reset)
        if models_info_path.stat().st_size == 0:
            return {"num_models": 0, "models": []}
    except Exception:
        pass
    
    try:
        with models_info_path.open('r') as f:
            content = f.read().strip()
            if not content:  # Empty file
                return {"num_models": 0, "models": []}
            precomputed_requirements = json.loads(content)
        
        # Convert precomputed_requirements format to display format
        for model_name, model_data in precomputed_requirements.items():
            # Format parameter count
            param_billion = model_data.get('model_p_billion')
            if param_billion is not None:
                if param_billion >= 1:
                    parameter_count = f"{int(param_billion)} Billion" if param_billion == int(param_billion) else f"{param_billion} Billion"
                else:
                    parameter_count = f"{param_billion * 1000:.0f} Million"
            else:
                parameter_count = "Not specified"
            
            # Format quantization
            quant_bits = model_data.get('quantization_bits')
            if quant_bits is not None:
                quantization = f"{quant_bits} bits"
            else:
                quantization = "Not specified"
            
            estimated_vram = model_data.get('required_vram_gb')
            if estimated_vram is None:
                estimated_vram = 0.0
            
            models.append({
                "name": model_data.get('model_name', model_name),
                "image": model_data.get('image', 'Not specified'),
                "image_size_gb": model_data.get('model_size_gb', 0.0) or 0.0,
                "parameter_count": parameter_count,
                "quantization": quantization,
                "estimated_vram_gb": estimated_vram,
                "supported_arch": model_data.get('supported_arch', 'unknown')
            })
        
        num_models = len(models)
        
    except Exception as e:
        st.error(f"Error loading model info from {models_info_path}: {str(e)}")
        return {"num_models": 0, "models": []}
    
    return {"num_models": num_models, "models": models}


def load_deployment_matrix():
    """Load deployment matrix entries from info/deployment_matrix.json."""
    matrix_path = _get_info_dir() / "deployment_matrix.json"

    if (
        st.session_state.agent_start_time is None
        or not matrix_path.exists()
        or matrix_path.stat().st_size == 0
    ):
        return []

    try:
        file_mtime = matrix_path.stat().st_mtime
        if file_mtime < st.session_state.agent_start_time:
            return []

        data = json.loads(matrix_path.read_text())
        if isinstance(data, dict):
            data = [data]
        if not isinstance(data, list):
            return []

        normalized = []
        for entry in data:
            if not isinstance(entry, dict):
                continue
            row = {
                "model_name": entry.get("model_name", "Unknown"),
                "deployable": bool(entry.get("deployable", False)),
                "reason": entry.get("reason", "No reason provided"),
            }
            if "post_remediation_ready" in entry:
                row["post_remediation_ready"] = bool(entry.get("post_remediation_ready"))
            normalized.append(row)
        return normalized
    except Exception as exc:
        st.error(f"Error loading deployment matrix: {exc}")
        return []


def _raw_models_info_map() -> dict:
    """Load precomputed models_info.json as a dict keyed by model name."""
    path = _get_info_dir() / "models_info.json"
    try:
        if not path.exists() or path.stat().st_size == 0:
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _model_has_persisted_serving_args(model_name: str) -> bool:
    """
    True if models_info lists non-empty serving args (from model-car serving_arguments.args).
    Empty or missing arguments means configuration is not ready for deployment.
    """
    m = _raw_models_info_map().get(model_name)
    if not isinstance(m, dict):
        return False
    args = m.get("arguments")
    if not isinstance(args, list):
        return False
    return len(args) > 0


def matrix_entry_fully_deployable(entry: dict) -> bool:
    """
    Stable rubric: matrix deployable flag, optional post_remediation_ready, and
    non-empty persisted serving arguments in models_info.json must all agree.
    """
    if not entry.get("deployable", False):
        return False
    if entry.get("post_remediation_ready") is False:
        return False
    mn = entry.get("model_name") or ""
    if mn and not _model_has_persisted_serving_args(mn):
        return False
    return True


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*m')

def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub('', text)


def stream_agent_command(cmd, env, cwd, live_output_placeholder, timeout_sec=2100, tail_lines=200):
    """Run a command and stream stdout/stderr into the UI while capturing full output."""
    import signal
    output_lines = []
    start_time = time.time()

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
        cwd=cwd,
        preexec_fn=os.setsid,
    )

    st.session_state.agent_pid = proc.pid

    assert proc.stdout is not None
    selector = selectors.DefaultSelector()
    selector.register(proc.stdout, selectors.EVENT_READ)

    if live_output_placeholder is not None:
        live_output_placeholder.code("Waiting for output...")

    interrupted = False
    try:
        while True:
            if st.session_state.get("agent_interrupted"):
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                    proc.wait(timeout=5)
                except Exception:
                    try:
                        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                    except Exception:
                        proc.kill()
                output_lines.append("\n--- Agent interrupted by user ---")
                interrupted = True
                break

            if time.time() - start_time > timeout_sec:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                raise subprocess.TimeoutExpired(cmd, timeout_sec, output="\n".join(output_lines))

            events = selector.select(timeout=0.1)
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                output_lines.append(_strip_ansi(line.rstrip("\n")))
                if live_output_placeholder is not None:
                    tail = "\n".join(output_lines[-tail_lines:])
                    live_output_placeholder.code(tail)

            if proc.poll() is not None:
                for line in proc.stdout:
                    output_lines.append(_strip_ansi(line.rstrip("\n")))
                if live_output_placeholder is not None and output_lines:
                    tail = "\n".join(output_lines[-tail_lines:])
                    live_output_placeholder.code(tail)
                break
    finally:
        selector.close()
        st.session_state.agent_pid = None

    rc = -1 if interrupted else (proc.returncode or 0)
    return rc, "\n".join(output_lines)

# Function to extract QA summary from agent output
def extract_qa_summary(agent_output: str) -> tuple[str, str]:
    """
    Extract QA validation summary from the full agent output.
    Returns: (status, message) tuple
    """
    if not agent_output:
        return "pending", "QA validation has not been run yet."
    
    # Look for QA Validation section
    qa_patterns = [
        r"###\s*QA\s*Validation\s*\n(.*?)(?=\n###|\Z)",
        r"##\s*QA\s*Validation\s*\n(.*?)(?=\n##|\Z)",
        r"QA\s*Validation\s*\n(.*?)(?=\n###|\n##|\Z)",
    ]
    
    qa_summary = None
    for pattern in qa_patterns:
        match = re.search(pattern, agent_output, re.IGNORECASE | re.DOTALL)
        if match:
            qa_summary = match.group(1).strip()
            break
    
    if not qa_summary:
        # Fallback: look for any mention of QA
        if "QA" in agent_output.upper() or "qa specialist" in agent_output.lower():
            # Try to find the last paragraph or sentence mentioning QA
            lines = agent_output.split('\n')
            qa_lines = []
            found_qa = False
            for line in lines:
                if 'qa' in line.lower() and ('specialist' in line.lower() or 'validation' in line.lower()):
                    found_qa = True
                if found_qa:
                    qa_lines.append(line)
                    # Stop at next section or after a few lines
                    if line.strip().startswith('###') and len(qa_lines) > 1:
                        qa_lines.pop()  # Remove the section header
                        break
                    if len(qa_lines) > 10:  # Limit to reasonable length
                        break
            if qa_lines:
                qa_summary = '\n'.join(qa_lines).strip()
    
    if not qa_summary:
        return "pending", "QA validation information not found in agent output."
    
    # Clean up the summary - remove excessive whitespace and logs
    # Remove lines that look like logs (contain timestamps, [QA], etc.)
    cleaned_lines = []
    for line in qa_summary.split('\n'):
        line = line.strip()
        # Skip log-like lines (timestamps, [QA] prefixes, command outputs, etc.)
        if (re.match(r'^\[.*\]', line) or 
            re.match(r'^\d{4}-\d{2}-\d{2}', line) or
            re.match(r'^\[QA\]', line) or
            line.startswith('Starting KServe deployment QA') or
            'INFO' in line or 'ERROR' in line or 'WARNING' in line):
            continue
        # Skip empty lines at start
        if not cleaned_lines and not line:
            continue
        # Stop at code blocks or JSON examples
        if line.startswith('```') or line.startswith('{') and 'json' in line.lower():
            break
        cleaned_lines.append(line)
    
    qa_summary = '\n'.join(cleaned_lines).strip()
    
    # Extract just the first paragraph or sentence if it's too long
    # Look for natural sentence boundaries
    if len(qa_summary) > 300:
        # Try to find the end of the first complete thought
        sentences = re.split(r'[.!?]\s+', qa_summary)
        if sentences:
            # Take first 2-3 sentences if they're reasonable length
            summary_parts = []
            total_len = 0
            for sent in sentences:
                if total_len + len(sent) > 300 and summary_parts:
                    break
                summary_parts.append(sent)
                total_len += len(sent) + 2
            if summary_parts:
                qa_summary = '. '.join(summary_parts)
                if not qa_summary.endswith(('.', '!', '?')):
                    qa_summary += '.'
    
    # Determine status from summary (fail before pass so "failed to …" does not match "success")
    qa_summary_lower = qa_summary.lower()
    if any(word in qa_summary_lower for word in ['not run', 'skipped', 'no-go', 'no go']):
        status = "skipped"
    elif any(word in qa_summary_lower for word in ['fail', 'error', 'failed', 'qa_error']):
        status = "failed"
    elif any(word in qa_summary_lower for word in ['pass', 'success', 'passed', 'completed successfully']):
        status = "passed"
    else:
        status = "completed"
    
    # Limit summary length to avoid showing full logs
    if len(qa_summary) > 500:
        # Try to find a good cutoff point (end of sentence)
        cutoff = qa_summary[:500].rfind('.')
        if cutoff > 200:
            qa_summary = qa_summary[:cutoff + 1]
        else:
            qa_summary = qa_summary[:500] + "..."
    
    return status, qa_summary


def extract_qa_model_progress(agent_output: str) -> list[dict]:
    """
    Parse QA_MODEL_* progress protocol emitted by the QA pipeline.
    Returns a list of model dicts: {model_name, isvc_name, status, detail, url}.
    """
    if not agent_output:
        return []

    models: dict[str, dict] = {}

    def ensure(name: str) -> dict:
        if name not in models:
            models[name] = {
                "model_name": name,
                "isvc_name": "",
                "status": "pending",
                "detail": "",
                "url": "",
            }
        return models[name]

    for raw in agent_output.splitlines():
        line = raw.strip()
        if not line.startswith("[QA]"):
            continue
        payload = line[len("[QA]") :].strip()
        if not payload.startswith("QA_MODEL_"):
            continue
        parts = payload.split("::")
        if len(parts) < 2:
            continue
        event = parts[0]
        model = parts[1] if len(parts) > 1 else ""
        isvc = parts[2] if len(parts) > 2 else ""

        if not model:
            continue
        rec = ensure(model)
        if isvc and isvc != "None":
            rec["isvc_name"] = isvc

        if event == "QA_MODEL_START":
            rec["status"] = "deploying"
        elif event == "QA_MODEL_APPLIED":
            rec["status"] = "applied"
        elif event == "QA_MODEL_HEAL":
            rec["status"] = "healing"
            rec["detail"] = "::".join(parts[2:]) if len(parts) > 2 else ""
        elif event == "QA_MODEL_READY":
            rec["status"] = "ready"
            # url is optional 4th field
            if len(parts) > 3:
                rec["url"] = parts[3].strip()
        elif event == "QA_MODEL_FAIL":
            rec["status"] = "failed"
            rec["detail"] = "::".join(parts[2:]) if len(parts) > 2 else ""

    # Stable ordering for UI display
    order = ["deploying", "applied", "healing", "ready", "failed", "pending"]
    return sorted(models.values(), key=lambda x: (order.index(x["status"]) if x["status"] in order else 99, x["model_name"]))

# Function to parse GPU info from gpu_info.txt
def parse_gpu_info():
    """Parse GPU information from info/gpu_info.txt file and return per-node details."""
    gpu_info_path = str(_get_info_dir() / "gpu_info.txt")
    gpu_data = {
        "nodes": [],  # List of individual node details
        "total_gpus": 0,
        "total_nodes": 0
    }
    
    if not os.path.exists(gpu_info_path):
        return gpu_data
    
    # Don't show data if agent hasn't started yet
    if st.session_state.agent_start_time is None:
        return gpu_data
    
    # Check if file has been updated since agent started (only show if new data)
    try:
        file_mtime = os.path.getmtime(gpu_info_path)
        file_size = os.path.getsize(gpu_info_path)
        # Only show if file was modified after agent started and is not empty
        if file_mtime < st.session_state.agent_start_time or file_size == 0:
            return gpu_data
    except Exception:
        pass
    
    try:
        with open(gpu_info_path, 'r') as f:
            content = f.read()
        
        # Split content by blank lines to get individual node entries
        entries = content.strip().split('\n\n')
        
        for entry in entries:
            entry = entry.strip()
            if not entry or 'Cloud Provider' not in entry:
                continue
            
            # Parse individual node
            node = {
                "cloud_provider": "Unknown",
                "instance_type": "Unknown",
                "gpu_provider": "Unknown",
                "gpu_product": "Unknown",
                "per_gpu_memory_gb": 0.0,
                "allocatable_gpus": 0,
                "node_ram_gb": 0.0,
                "node_storage_gb": 0.0,
                "node_name": "Unknown"
            }
            
            lines = entry.split('\n')
            for line in lines:
                line = line.strip()
                if line and '•' in line:
                    parts = line.replace('•', '').strip().split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        
                        if key == "Cloud Provider":
                            node["cloud_provider"] = value
                        elif key == "Instance Type":
                            node["instance_type"] = value
                        elif key == "GPU Provider":
                            node["gpu_provider"] = value
                        elif key == "GPU Product":
                            node["gpu_product"] = value
                        elif key == "Per-GPU Memory":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["per_gpu_memory_gb"] = float(match.group(1))
                        elif key == "Allocatable GPUs":
                            try:
                                node["allocatable_gpus"] = int(value)
                            except ValueError:
                                pass
                        elif key == "Node RAM":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["node_ram_gb"] = float(match.group(1))
                        elif key == "Node Storage":
                            match = re.search(r'(\d+\.?\d*)', value)
                            if match:
                                node["node_storage_gb"] = float(match.group(1))
                        elif key == "Node Name":
                            node["node_name"] = value
            
            # Add node to list if it has valid data
            if node["node_name"] != "Unknown" or node["allocatable_gpus"] > 0:
                gpu_data["nodes"].append(node)
                gpu_data["total_gpus"] += node["allocatable_gpus"]
        
        gpu_data["total_nodes"] = len(gpu_data["nodes"])
        
    except Exception as e:
        st.error(f"Error parsing GPU info: {str(e)}")
    
    return gpu_data

# Sidebar for API key and YAML upload
with st.sidebar:
    st.title("Configuration")
    
    # Gemini API Key input (mandatory)
    st.subheader("Gemini API Key *")
    api_key_input = st.text_input(
        "Enter your Gemini API Key",
        type="password",
        value=st.session_state.gemini_api_key or "",
        placeholder="Enter your Gemini API Key",
        help="Get your API key to Run the Agent (required)"
    )
    
    if api_key_input:
        st.session_state.gemini_api_key = api_key_input
        os.environ["GEMINI_API_KEY"] = api_key_input
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Configured</span>', unsafe_allow_html=True)
    
    
    # OCI Registry Pull Secret input (mandatory)
    st.subheader("OCI Registry Pull Secret *")
    oci_secret_input = st.text_input(
        "Enter your OCI Registry Pull Secret",
        type="password",
        value=st.session_state.oci_pull_secret or "",
        help="Enter your OCI registry pull secret for authentication (required)"
    )
    
    if oci_secret_input:
        st.session_state.oci_pull_secret = oci_secret_input
        os.environ["OCI_REGISTRY_PULL_SECRET"] = oci_secret_input
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Configured</span>', unsafe_allow_html=True)
    
    # Registry Host input (optional)
    st.subheader("Registry Host")
    registry_host_input = st.text_input(
        "Enter Registry Host",
        value=st.session_state.registry_host or "",
        placeholder="e.g., registry.redhat.io",
        help="Override the registry host for QA tests (optional). If blank, it is auto-detected from the model-car YAML images."
    )
    
    if registry_host_input and registry_host_input.strip():
        st.session_state.registry_host = registry_host_input.strip()
        os.environ["REGISTRY_HOST"] = registry_host_input.strip()
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Configured</span>', unsafe_allow_html=True)
    else:
        st.session_state.registry_host = None
    
    # vLLM Runtime Image input (optional)
    st.subheader("vLLM Runtime Image")
    # Check environment variable if session state is not set
    default_vllm_image = st.session_state.vllm_runtime_image or os.environ.get("VLLM_RUNTIME_IMAGE", "")
    vllm_image_input = st.text_input(
        "Enter vLLM Runtime Image",
        value=default_vllm_image,
        placeholder="e.g., quay.io/opendatahub/vllm-runtime:latest",
        help="Specify the vLLM runtime image to use for deployment (optional). Can also be set via VLLM_RUNTIME_IMAGE environment variable."
    )
    
    if vllm_image_input:
        st.session_state.vllm_runtime_image = vllm_image_input
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Configured</span>', unsafe_allow_html=True)
    
    # Runtime Accelerator dropdown (optional)
    st.subheader("Runtime Accelerator")
    runtime_backend_options = [
        "Nvidia - CUDA",
        "AMD - ROCm",
        "Intel-Gaudi",
        "IBM Spyre - Spyre"
    ]
    
    # Get current index for selectbox
    current_index = None  # Default to placeholder
    if st.session_state.runtime_backend and st.session_state.runtime_backend in runtime_backend_options:
        current_index = runtime_backend_options.index(st.session_state.runtime_backend)
    
    runtime_backend_selected = st.selectbox(
        "Select Runtime Accelerator",
        options=runtime_backend_options,
        index=current_index,
        placeholder="Select an option...",
        help="Select the runtime accelerator for vLLM deployment (optional)"
    )
    
    if runtime_backend_selected:
        st.session_state.runtime_backend = runtime_backend_selected
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Selected</span>', unsafe_allow_html=True)
    else:
        st.session_state.runtime_backend = None
    
    # oc login command (optional)
    st.subheader("oc login Command")
    oc_login_input = st.text_input(
        "Paste your oc login command",
        type="password",
        value=st.session_state.oc_login_command or "",
        placeholder="oc login --token=sha256~... --server=https://...",
        help="Paste the full oc login command from the OpenShift console (optional). The agent will execute it before running."
    )

    if oc_login_input and oc_login_input.strip():
        st.session_state.oc_login_command = oc_login_input.strip()
        st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Configured</span>', unsafe_allow_html=True)
    else:
        st.session_state.oc_login_command = None

    st.subheader("QA cleanup")
    st.session_state.qa_skip_namespace_delete = st.checkbox(
        "Keep model-validation namespace after successful QA",
        value=st.session_state.qa_skip_namespace_delete,
        help=(
            "When checked, sets QA_SKIP_NAMESPACE_DELETE=1 for the agent process: the "
            "`model-validation` namespace is not deleted after all models pass (useful for debugging)."
        ),
    )

    # YAML file upload (mandatory)
    st.subheader("Upload Modelcar Images YAML File *")
    uploaded_file = st.file_uploader(
        "Choose a YAML file (required)",
        type=['yaml', 'yml'],
        help="Upload your Modelcar Images YAML file (required)"
    )
    
    if uploaded_file is not None:
        try:
            yaml_content = uploaded_file.read()
            st.session_state.yaml_config = yaml.safe_load(yaml_content)
            # Store original YAML content to preserve exact format
            st.session_state.yaml_content_raw = yaml_content
            st.markdown('<span style="background-color: #e6fcf5; color: #00b894; padding: 4px 10px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Loaded</span>', unsafe_allow_html=True)
            
            # Display YAML content in expander
            with st.expander("View YAML Configuration"):
                st.code(yaml_content.decode('utf-8'), language='yaml')
        except yaml.YAMLError as e:
            st.error(f"Error parsing YAML file: {str(e)}")
        except Exception as e:
            st.error(f"Error reading file: {str(e)}")
    
    st.markdown('<hr style="margin:0.5rem 0;border:none;border-top:1px solid #dfe6e9;">', unsafe_allow_html=True)
    
    # Reset button
    if st.button("Reset", width='stretch'):
        old_dir = st.session_state.pop("agent_run_info_dir", None)
        if isinstance(old_dir, Path) and old_dir.is_dir():
            shutil.rmtree(old_dir, ignore_errors=True)
        st.session_state.agent_run_info_dir = None

        # Reset session state
        st.session_state.agent_started = False
        st.session_state.workflow_completed = False
        st.session_state.workflow_step = 0
        st.session_state.start_time = None
        st.session_state.agent_result = None
        st.session_state.agent_output_text = None
        st.session_state.agent_command_output = None
        st.session_state.agent_start_time = None
        st.session_state.yaml_content_raw = None
        st.session_state.config_path = None
        st.session_state.preflight_results = None
        st.session_state.oc_login_command = None
        st.session_state.agent_timestamps = {
            "supervisor": None,
            "configuration": None,
            "accelerator": None,
            "deployment": None,
            "qa": None
        }
        st.session_state.agent_interrupted = False
        st.session_state.agent_pid = None
        st.session_state.qa_skip_namespace_delete = False
        st.rerun()

# Main interface — report-style header banner
_header_logo_b64 = _load_logo_b64()
_header_logo_html = (
    f'<img src="data:image/png;base64,{_header_logo_b64}" alt="OpenShift AI" '
    f'style="height:120px; flex-shrink:0;">'
    if _header_logo_b64 else ""
)
_github_svg = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 16 16" fill="white" style="vertical-align:middle; margin-right:5px;">'
    '<path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
    '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53'
    '.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 '
    '0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 '
    '1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 '
    '3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.01 '
    '8.01 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
)
st.markdown(f"""
<div style="
    background: #ffffff;
    color: #1e2a38;
    padding: 12px 32px 4px 32px;
    border-radius: 0px;
    margin: 0 0 2px 0;
    display: flex;
    align-items: center;
    gap: 18px;
    flex-wrap: wrap;
    font-family: 'Red Hat Display', -apple-system, BlinkMacSystemFont, sans-serif;
    border: 1px solid #dfe6e9;
    border-left: 5px solid #e63946;
    border-top: 1px solid #dfe6e9;
    box-shadow: 0 -1px 4px rgba(0,0,0,0.04), 0 2px 8px rgba(0,0,0,0.06);
">
    {_header_logo_html}
    <h1 style="margin:0; font-size:1.55rem; color:#1e2a38; font-weight:700;">
        Model Runtimes Agent &mdash; OpenShift AI
    </h1>
    <a href="https://github.com/opendatahub-io/model-runtimes-agent" target="_blank" style="text-decoration:none; margin-left:auto;">
        <span style="background:#24292f; color:#fff; padding:6px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; letter-spacing:0.3px; display:inline-flex; align-items:center;">
            {_github_svg} GitHub Repo
        </span>
    </a>
    <a href="#" target="_blank" style="text-decoration:none;">
        <span style="background:#e63946; color:#fff; padding:6px 14px; border-radius:20px; font-size:0.82rem; font-weight:600; letter-spacing:0.3px;">
            Model Runtimes in RHOAI
        </span>
    </a>
</div>
""", unsafe_allow_html=True)



# Start AI Agent button
if not st.session_state.agent_started:
    # Show pre-flight checks before agent starts
    st.subheader("System Checks")
    if st.session_state.preflight_results is None:
        with st.spinner("Checking dependencies..."):
            results = run_preflight_checks(quiet=True)
            st.session_state.preflight_results = [
                {
                    "name": r.name,
                    "installed": r.installed,
                    "version": r.version,
                    "path": r.path,
                    **({"running": r.running} if r.running is not None else {}),
                    **({"running_detail": r.running_detail} if r.running_detail else {}),
                }
                for r in results
            ]
    _preflight_badges = ""
    for r in st.session_state.preflight_results:
        podman_down = (
            r.get("name") == "podman"
            and r.get("installed")
            and r.get("running") is False
        )
        name_esc = html.escape(str(r.get("name", "")), quote=False)
        ver_raw = r.get("version") or ""
        ver_esc = f" ({html.escape(str(ver_raw), quote=False)})" if ver_raw else ""
        if r["installed"] and not podman_down:
            extra = ""
            if r.get("name") == "podman" and r.get("running") is True:
                extra = " · engine OK"
            _preflight_badges += (
                f'<span style="background-color: #00b894; color: white; padding: 5px 14px; border-radius: 20px; '
                f'font-size: 0.82rem; font-weight:600; margin-right: 8px; display:inline-block;">'
                f'{name_esc} &#10003;{ver_esc}{extra}</span>'
            )
        elif podman_down:
            hint = html.escape(
                r.get("running_detail")
                or "Start Podman (e.g. podman machine start on macOS).",
                quote=True,
            )
            _preflight_badges += (
                f'<span style="background-color: #e17055; color: white; padding: 5px 14px; border-radius: 20px; '
                f'font-size: 0.82rem; font-weight:600; margin-right: 8px; display:inline-block;" '
                f'title="{hint}">'
                f'{name_esc} &#10007; INSTALLED · ENGINE DOWN{ver_esc}</span>'
            )
        else:
            _preflight_badges += (
                f'<span style="background-color: #d63031; color: white; padding: 5px 14px; border-radius: 20px; '
                f'font-size: 0.82rem; font-weight:600; margin-right: 8px; display:inline-block;">'
                f'{name_esc} &#10007; NOT FOUND</span>'
            )
    st.markdown(f'<div style="display:flex; flex-wrap:wrap; gap:6px; align-items:center;">{_preflight_badges}</div>', unsafe_allow_html=True)

    st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 2, 3])
    with col2:
        if st.button("Start Runtime Agent", type="primary"):
            if not st.session_state.gemini_api_key:
                st.error("Please enter your Gemini API key in the sidebar first!")
            elif not st.session_state.oci_pull_secret:
                st.error("Please enter your OCI Registry Pull Secret in the sidebar first!")
            elif not st.session_state.yaml_config:
                st.error("Please upload a YAML configuration file in the sidebar first!")
            else:
                try:
                    if st.session_state.gemini_api_key:
                        os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
                    if st.session_state.oci_pull_secret:
                        os.environ["OCI_REGISTRY_PULL_SECRET"] = st.session_state.oci_pull_secret

                    _ensure_run_info_dir()

                    temp_dir = tempfile.gettempdir()
                    config_path = os.path.join(temp_dir, "modelcar_config.yaml")
                    with open(config_path, 'wb') as tmp_file:
                        if hasattr(st.session_state, 'yaml_content_raw') and st.session_state.yaml_content_raw:
                            tmp_file.write(st.session_state.yaml_content_raw)
                        else:
                            yaml.dump(st.session_state.yaml_config, tmp_file)
                    
                    st.session_state.config_path = config_path
                    st.session_state.agent_started = True
                    st.session_state.agent_interrupted = False
                    st.session_state.agent_pid = None
                    st.session_state.start_time = time.time()
                    st.session_state.workflow_step = 1
                    
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to initialize: {str(e)}")
                    st.exception(e)
else:
    # Status checks
    st.subheader("Status Checks")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.gemini_api_key:
            st.markdown('<span style="background-color: #00b894; color: white; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">Gemini API Key: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #d63031; color: white; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">Gemini API Key: Not configured</span>', unsafe_allow_html=True)
    
    with col2:
        if st.session_state.oci_pull_secret:
            st.markdown('<span style="background-color: #00b894; color: white; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">OCI Pull Secret: Verified</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span style="background-color: #d63031; color: white; padding: 6px 16px; border-radius: 20px; font-size: 0.85rem; font-weight: 600;">OCI Pull Secret: Not configured</span>', unsafe_allow_html=True)
    
    
    # Workflow progress
    st.subheader("Agent Workflow Progress")
    
    _ts = st.session_state.agent_timestamps
    _agent_started = st.session_state.agent_start_time is not None

    workflow_steps = [
        {"name": "Starting Supervisor Agent",    "status": "completed" if _agent_started else "pending"},
        {"name": "Calling Configuration Agent",  "status": "completed" if _ts.get("configuration") else ("in_progress" if _agent_started and not _ts.get("configuration") else "pending")},
        {"name": "Calling Accelerator Agent",    "status": "completed" if _ts.get("accelerator") else ("in_progress" if _ts.get("configuration") and not _ts.get("accelerator") else "pending")},
        {"name": "Calling Deployment Specialist","status": "completed" if _ts.get("deployment") else ("in_progress" if _ts.get("accelerator") and not _ts.get("deployment") else "pending")},
        {"name": "Calling QA Specialist",        "status": "completed" if _ts.get("supervisor") else ("in_progress" if _ts.get("deployment") and not _ts.get("supervisor") else "pending")},
    ]

    completed_count = sum(1 for s in workflow_steps if s["status"] == "completed")
    progress_value = max(0.0, min(1.0, completed_count / len(workflow_steps)))
    st.progress(progress_value)
    
    # Display step statuses with badges
    for i, step in enumerate(workflow_steps):
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"{step['name']}")
        with col2:
            if step["status"] == "completed":
                st.markdown('<span style="background-color: #00b894; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Completed</span>', unsafe_allow_html=True)
            elif step["status"] == "in_progress":
                st.markdown('<span style="background-color: #0984e3; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">In Progress</span>', unsafe_allow_html=True)
            else:
                st.markdown('<span style="background-color: #636e72; color: white; padding: 5px 14px; border-radius: 20px; font-size: 0.82rem; font-weight:600;">Pending</span>', unsafe_allow_html=True)

    _rerun_busy = _agent_subprocess_is_running()
    _rerun_cols = st.columns([2, 2, 6])
    with _rerun_cols[0]:
        if st.button(
            "Rerun with same configuration",
            help="Runs the supervisor again using the current sidebar API key, OCI pull secret, YAML, and optional oc login — without clearing those fields.",
            disabled=_rerun_busy,
            key="rerun_same_config_btn",
        ):
            try:
                _prepare_rerun_from_session()
                st.rerun()
            except ValueError as exc:
                st.error(str(exc))
    with _rerun_cols[1]:
        if _rerun_busy:
            st.caption("Finish or stop the current run to enable Rerun.")

    st.markdown("---")
    
    # Show loader/spinner above all outputs when agent is running
    if st.session_state.workflow_step >= 1 and st.session_state.workflow_step < 6 and not st.session_state.workflow_completed:
        with st.spinner("Running supervisor agent..."):
            time.sleep(0.1)  # Small delay to show spinner
        st.markdown("")  # Add spacing after spinner

    live_output_placeholder = None
    if st.session_state.workflow_step == 1 and not st.session_state.workflow_completed:
        st.subheader("Live Agent Output")
        st.caption("Streaming the last 200 lines while the agent runs.")
        live_output_placeholder = st.empty()
        if st.button("Stop Agent", type="secondary", key="stop_agent_btn"):
            st.session_state.agent_interrupted = True
            st.warning("Interrupting agent process...")
    
    # Display outputs based on workflow step
    # Only show Configuration Agent Results if agent has started
    if st.session_state.workflow_step >= 1 and st.session_state.agent_start_time is not None:
        st.subheader("Configuration Agent Results")
        
        # Load model information from info/models_info.json
        model_info = load_model_info_from_json()
        num_models = model_info["num_models"]
        models = model_info["models"]
        
        # Only display if we have models (file was updated)
        if num_models > 0:
            # Track configuration agent completion time
            if st.session_state.agent_timestamps["configuration"] is None:
                st.session_state.agent_timestamps["configuration"] = time.time()
            
            # Display number of models at the top
            st.markdown(f"""
            <div style="border: 1px solid #dfe6e9; border-radius: 8px; padding: 16px; margin-bottom: 16px; background-color: #f5f6fa;">
            <p style="font-size: 1.15em; font-weight: 700; color: #00b894; font-family: 'Red Hat Display', sans-serif;">Number of models found in Modelcar YAML: {num_models}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display all model details in a single expandable dropdown
            if models:
                with st.expander("Model Details", expanded=False):
                    for idx, model in enumerate(models, 1):
                        st.markdown(f"""
                        <div style="padding: 18px; margin-bottom: 16px; background-color: #f5f6fa; border-radius: 8px; border: 1px solid #dfe6e9; border-left: 4px solid #0984e3;">
                        <h4 style="margin-top: 0; margin-bottom: 12px; color: #0984e3; font-family: 'Red Hat Display', sans-serif; font-size: 0.95rem;">Model {idx}: {model.get('name', 'Unknown')}</h4>
                        <ul style="list-style-type: none; padding-left: 0; margin-bottom: 0; font-size: 0.88rem; line-height: 1.8;">
                        <li><strong>Model Name</strong>: <code style="background:#eef1f5; padding:2px 6px; border-radius:4px; font-size:0.82rem; color:#d63031;">{model.get('name', 'Unknown')}</code></li>
                        <li><strong>Image</strong>: {model.get('image', 'Not specified')}</li>
                        <li><strong>Image Size</strong>: {model.get('image_size_gb', 0.0)} GB</li>
                        <li><strong>Parameter Count</strong>: {model.get('parameter_count', 'Not specified')}</li>
                        <li><strong>Quantization</strong>: {model.get('quantization', 'Not specified')}</li>
                        <li><strong>Estimated VRAM</strong>: {model.get('estimated_vram_gb', 0.0)} GB</li>
                        <li><strong>Supported Architectures</strong>: {model.get('supported_arch', 'unknown')}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            if st.session_state.workflow_step >= 2:
                st.info("Configuration Agent is processing. Please wait...")
        
        st.markdown("---")
    
    if st.session_state.workflow_step >= 2:
        st.subheader("Accelerator Agent Results")
        
        # Fetch latest GPU info from gpu_info.txt
        gpu_info = parse_gpu_info()
        
        # Only display if we have GPU nodes (file was updated)
        if gpu_info["total_nodes"] > 0:
            # Track accelerator agent completion time
            if st.session_state.agent_timestamps["accelerator"] is None:
                st.session_state.agent_timestamps["accelerator"] = time.time()
            
            # Display total number of GPU nodes
            total_nodes = gpu_info["total_nodes"]
            total_gpus = gpu_info["total_gpus"]
            
            st.markdown(f"""
            <div style="border: 1px solid #dfe6e9; border-radius: 8px; padding: 20px; margin-bottom: 16px; background-color: #f5f6fa;">
            <p style="font-size: 1.15em; font-weight: 700; color: #00b894; margin-bottom: 6px;">Total Number of GPU Nodes: {total_nodes}</p>
            <p style="font-size: 1.05em; font-weight: 700; color: #00b894; margin-bottom: 0;">Total Number of GPUs Available: {total_gpus}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Display details for each GPU node in expandable sections
            if gpu_info["nodes"]:
                for idx, node in enumerate(gpu_info["nodes"], 1):
                    accelerator_status = "GPU available" if node["allocatable_gpus"] > 0 else "No GPU available"
                    
                    with st.expander(f"GPU Node {idx} Details - {node['node_name']}", expanded=False):
                        st.markdown(f"""
                        <div style="padding: 18px; background-color: #f5f6fa; border-radius: 8px; border: 1px solid #dfe6e9;">
                        <ul style="list-style-type: none; padding-left: 0; font-size: 0.88rem; line-height: 1.8;">
                        <li><strong>Node Name</strong>: {node['node_name']}</li>
                        <li><strong>Status</strong>: {accelerator_status}</li>
                        <li><strong>Cloud Provider</strong>: {node['cloud_provider']}</li>
                        <li><strong>Instance Type</strong>: {node['instance_type']}</li>
                        <li><strong>GPU Provider</strong>: {node['gpu_provider']}</li>
                        <li><strong>GPU Product</strong>: {node['gpu_product']}</li>
                        <li><strong>Per-GPU Memory</strong>: {node['per_gpu_memory_gb']} GB</li>
                        <li><strong>Allocatable GPUs</strong>: {node['allocatable_gpus']}</li>
                        <li><strong>Node RAM</strong>: {node['node_ram_gb']} GB</li>
                        <li><strong>Node Storage</strong>: {node['node_storage_gb']} GB</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.info("Accelerator Agent is processing. Please wait...")
        
        st.markdown("---")
    
    if st.session_state.workflow_step >= 3:
        st.subheader("Deployment Specialist Results")
        
        # Load deployment info from info/deployment_info.txt
        deployment_info_path = str(_get_info_dir() / "deployment_info.txt")
        
        # Check if file exists and has been updated since agent started
        file_exists_and_updated = False
        if os.path.exists(deployment_info_path) and st.session_state.agent_start_time is not None:
            try:
                file_mtime = os.path.getmtime(deployment_info_path)
                file_size = os.path.getsize(deployment_info_path)
                # Only show if file was modified after agent started and is not empty
                if file_mtime >= st.session_state.agent_start_time and file_size > 0:
                    file_exists_and_updated = True
            except Exception:
                pass
        
        if file_exists_and_updated:
            # Track deployment agent completion time
            if st.session_state.agent_timestamps["deployment"] is None:
                st.session_state.agent_timestamps["deployment"] = time.time()
            
            try:
                with open(deployment_info_path, 'r', encoding='utf-8') as f:
                    deployment_info_text = f.read().strip()
                
                # Extract verdict for card and expander label from deployment_info.txt
                # Parse the actual format from the file (e.g., "### Deployment Decision: NO-GO" or "**Verdict: GO**")
                verdict = "GO"
                verdict_color = "#00b894"
                
                deployment_info_upper = deployment_info_text.upper()
                if any(pattern in deployment_info_upper for pattern in [
                    "DEPLOYMENT DECISION: NO-GO",
                    "DEPLOYMENT DECISION: NO GO",
                    "VERDICT: NO-GO",
                    "VERDICT: NO GO",
                    "**VERDICT: NO-GO**",
                    "**VERDICT: NO GO**"
                ]):
                    verdict = "NO-GO"
                    verdict_color = "#d63031"
                elif any(pattern in deployment_info_upper for pattern in [
                    "DEPLOYMENT DECISION: GO",
                    "VERDICT: GO",
                    "**VERDICT: GO**"
                ]):
                    verdict = "GO"
                    verdict_color = "#00b894"
                
                # Display in expandable dropdown
                with st.expander(f"Deployment Decision Details", expanded=False):
                    # Render the markdown content using Streamlit's native markdown renderer
                    st.markdown(deployment_info_text)
            except Exception as e:
                st.error(f"Error reading deployment info from {deployment_info_path}: {str(e)}")
        else:
            st.info("Deployment Specialist is processing. Please wait...")
        
        st.markdown("---")

    deployment_matrix_entries = load_deployment_matrix()
    if deployment_matrix_entries:
        st.subheader("Deployment Matrix")
        deployable = [entry for entry in deployment_matrix_entries if matrix_entry_fully_deployable(entry)]
        blocked = [entry for entry in deployment_matrix_entries if not matrix_entry_fully_deployable(entry)]

        col_deployable, col_blocked = st.columns(2)
        with col_deployable:
            st.markdown("**Deployable**")
            if deployable:
                for entry in deployable:
                    st.markdown(
                        f'<div style="background:#e6fcf5; border-left:4px solid #00b894; padding:12px 16px; border-radius:8px; margin-bottom:8px; font-size:0.88rem;">'
                        f'&#10003; <strong>{entry["model_name"]}</strong><br>'
                        f'<small style="color:#636e72;">{entry["reason"]}</small></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_None_")

        with col_blocked:
            st.markdown("**Non-deployable**")
            if blocked:
                for entry in blocked:
                    st.markdown(
                        f'<div style="background:#ffeaea; border-left:4px solid #d63031; padding:12px 16px; border-radius:8px; margin-bottom:8px; font-size:0.88rem;">'
                        f'&#10007; <strong>{entry["model_name"]}</strong><br>'
                        f'<small style="color:#636e72;">{entry["reason"]}</small></div>',
                        unsafe_allow_html=True,
                    )
            else:
                st.markdown("_None_")

        st.markdown("---")
    
    if st.session_state.workflow_step >= 4:
        st.subheader("QA Specialist Results")
        
        # Extract QA summary from full agent output
        agent_output = st.session_state.agent_output_text or ""
        qa_status, qa_message = extract_qa_summary(agent_output)
        qa_models = extract_qa_model_progress(agent_output)
        
        # If no QA info found and agent output exists, show a message
        if qa_status == "pending" and agent_output:
            qa_message = "QA validation information is being processed. Please check the full agent output for details."
        
        if qa_status == "passed" or qa_status == "completed":
            status_color = "#00b894"
        elif qa_status == "failed":
            status_color = "#d63031"
        elif qa_status == "skipped":
            status_color = "#e17055"
        else:
            status_color = "#636e72"
        
        st.markdown(f"""
        <div style="border: 1px solid #dfe6e9; border-radius: 8px; padding: 20px; max-height: 300px; overflow-y: auto; background-color: #f5f6fa;">
        <p><strong>Status:</strong> <span style="color: {status_color}; font-weight: 700;">{qa_status.upper()}</span></p>
        <p style="font-size: 0.88rem; line-height: 1.7; color: #2d3436;">{qa_message}</p>
        </div>
        """, unsafe_allow_html=True)

        if qa_models:
            st.markdown("<div style='margin-top: 12px;'></div>", unsafe_allow_html=True)
            st.markdown("**Per-model deployment progress**")
            for m in qa_models:
                status = m.get("status", "pending")
                name = m.get("model_name", "")
                isvc = m.get("isvc_name", "")
                detail = m.get("detail", "")
                url = m.get("url", "")

                color = "#636e72"
                bg = "#f1f2f6"
                mark = "…"
                if status == "ready":
                    color, bg, mark = "#00b894", "#e6fcf5", "✓"
                elif status in ("deploying", "applied", "healing"):
                    color, bg, mark = "#0984e3", "#e8f4ff", "⟳"
                elif status == "failed":
                    color, bg, mark = "#d63031", "#ffeaea", "✗"

                extra = []
                if isvc:
                    extra.append(f"ISVC: {isvc}")
                if url:
                    extra.append(f"URL: {url}")
                if detail:
                    extra.append(detail)
                extra_html = "<br>".join(f"<small style='color:#636e72;'>{e}</small>" for e in extra)

                st.markdown(
                    f"""
<div style="background:{bg}; border-left:4px solid {color}; padding:10px 14px; border-radius:8px; margin-bottom:8px; font-size:0.88rem;">
  {mark} <strong>{name}</strong> <span style="color:{color}; font-weight:700;">({status})</span>
  {('<br>' + extra_html) if extra_html else ''}
</div>
""",
                    unsafe_allow_html=True,
                )
        st.markdown("---")
    
    # if st.session_state.workflow_step >= 5:
    #     st.subheader("Reporting Agent Results")
    #     elapsed_time = time.time() - st.session_state.start_time
    #     models_executed = get_value("reporting.models_executed", 2)
    #     reporting_status = get_value("reporting.status", "Deployment completed successfully")
    #     reporting_channel = get_value("reporting.channel", "#deployment_agent_report")
    #     
    #     st.markdown(f"""
    #     <div style="border: 1px solid #ddd; border-radius: 8px; padding: 16px; max-height: 300px; overflow-y: auto; background-color: #f8f9fa;">
    #     <h3>Slack Notification</h3>
    #     <p>The <strong>Reporting Agent</strong> has successfully sent the deployment summary to Slack channel <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;"><strong>{reporting_channel}</strong></a>.</p>
    #     <p><strong>Summary sent:</strong></p>
    #     <ul>
    #     <li><strong>Models Executed</strong>: {models_executed}</li>
    #     <li><strong>Time Taken</strong>: {elapsed_time:.2f} seconds</li>
    #     <li><strong>Status</strong>: {reporting_status}</li>
    #     <li><strong>Channel</strong>: <a href="https://slack.com" target="_blank" style="color: #007bff; text-decoration: none; font-weight: bold;">{reporting_channel}</a></li>
    #     </ul>
    #     <p style="margin-top: 12px; color: #28a745;"><strong>✓ Message delivered successfully</strong></p>
    #     </div>
    #     """, unsafe_allow_html=True)
    #     st.markdown("---")
    
    if st.session_state.workflow_step >= 5:
        st.subheader("Summary")
        elapsed_time = time.time() - st.session_state.start_time
        agent_out_summary = st.session_state.agent_output_text or ""
        qa_status_summary, _ = extract_qa_summary(agent_out_summary)
        # Count models that pass matrix + post_remediation + persisted serving args (models_info.json)
        deployment_matrix_entries = load_deployment_matrix()
        deployable_models = [entry for entry in deployment_matrix_entries if matrix_entry_fully_deployable(entry)]
        models_executed_summary = len(deployable_models) if deployable_models else 0
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Models Executed", str(models_executed_summary))
        with col2:
            elapsed_minutes = elapsed_time / 60
            st.metric("Time Taken", f"{elapsed_minutes:.2f} minutes")
        
        qa_ok_badge = deployment_success_badge_ok(agent_out_summary, qa_status_summary)
        if qa_ok_badge:
            st.markdown(
                '<div style="margin-top: 20px;"><span style="background-color: #00b894; color: white; padding: 8px 20px; border-radius: 20px; font-size: 0.88rem; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase;">Deployment completed successfully</span></div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div style="margin-top: 20px; width: 100%; box-sizing: border-box; padding: 18px 22px; border-radius: 12px; border: 1px solid #fadbd8;
                border-left: 4px solid #e17055; background: linear-gradient(135deg, #fffbf9 0%, #f8f9fa 100%);
                box-shadow: 0 2px 8px rgba(30, 42, 56, 0.06);">
                <div style="display: flex; align-items: flex-start; gap: 16px;">
                <span style="flex-shrink: 0; display: inline-flex; align-items: center; justify-content: center;
                width: 36px; height: 36px; border-radius: 50%; background: #fde8e4; color: #c0392b;
                font-weight: 800; font-size: 1rem; font-family: 'Red Hat Display', sans-serif;" aria-hidden="true">!</span>
                <div style="flex: 1; min-width: 0;">
                <div style="font-weight: 700; color: #c0392b; font-size: 1rem; letter-spacing: -0.02em;
                font-family: 'Red Hat Display', sans-serif;">QA did not complete successfully</div>
                </div></div></div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div style="margin-top: 1.5rem;"></div>', unsafe_allow_html=True)

        # Generate HTML report to the run info dir (View Report opens this file in the browser)
        try:
            info_dir = _get_info_dir()
            info_dir.mkdir(parents=True, exist_ok=True)
            report_path = info_dir / "ui_report.html"
            generate_html_report(
                info_dir=info_dir,
                output_path=report_path,
                agent_output=st.session_state.agent_output_text,
                preflight_results=st.session_state.preflight_results,
            )
            st.session_state["last_html_report_path"] = str(report_path.resolve())
            report_html = report_path.read_text(encoding="utf-8")
            # All three actions on one row (tight horizontal gap)
            col_dl, col_view, col_rerun = st.columns(3, gap="xxsmall")
            with col_dl:
                st.download_button(
                    label="Download HTML Report",
                    data=report_html,
                    file_name="report.html",
                    mime="text/html",
                    use_container_width=True,
                )
            with col_view:
                if st.button(
                    "View Report",
                    help="Opens the HTML report in your default browser (local Streamlit only).",
                    use_container_width=True,
                    key="view_report_summary_btn",
                ):
                    rp = st.session_state.get("last_html_report_path")
                    if rp and Path(rp).is_file():
                        webbrowser.open(Path(rp).resolve().as_uri())
                    else:
                        st.warning("Report file is not available yet.")
            with col_rerun:
                if st.button(
                    "Rerun",
                    help="Run the supervisor again with the same sidebar inputs (new artifact directory).",
                    disabled=_agent_subprocess_is_running(),
                    key="rerun_from_summary_btn",
                    use_container_width=True,
                ):
                    try:
                        _prepare_rerun_from_session()
                        st.rerun()
                    except ValueError as exc:
                        st.error(str(exc))
        except Exception as report_exc:
            st.warning(f"Could not generate report: {report_exc}")
        
        # Interactive Graphs
        st.markdown("---")
        st.subheader("Deployment Analytics")
        
        # Create tabs for different visualizations
        tab1, tab2 = st.tabs(["Agent Execution Timeline", "Resource Usage"])
        
        with tab1:
            agent_timestamps = st.session_state.agent_timestamps
            agent_start = st.session_state.agent_start_time if st.session_state.agent_start_time else time.time() - elapsed_time

            ordered_keys = [
                ("configuration", "Configuration"),
                ("accelerator", "Accelerator"),
                ("deployment", "Deployment"),
                ("supervisor", "Supervisor (Total)"),
            ]

            prev_end = agent_start
            agents_data = []
            for key, label in ordered_keys:
                ts = agent_timestamps.get(key)
                if not ts:
                    continue
                if key == "supervisor":
                    duration = ts - agent_start
                    start_offset = 0.0
                else:
                    duration = max(0, ts - prev_end)
                    start_offset = prev_end - agent_start
                    prev_end = ts
                agents_data.append({
                    'Agent': label,
                    'Start': start_offset,
                    'Duration': duration,
                })

            if not agents_data:
                agents_data = [
                    {'Agent': 'Configuration', 'Start': 0, 'Duration': elapsed_time * 0.25},
                    {'Agent': 'Accelerator', 'Start': elapsed_time * 0.25, 'Duration': elapsed_time * 0.25},
                    {'Agent': 'Deployment', 'Start': elapsed_time * 0.5, 'Duration': elapsed_time * 0.25},
                    {'Agent': 'Supervisor (Total)', 'Start': 0, 'Duration': elapsed_time},
                ]

            agent_data = pd.DataFrame(agents_data)

            fig_timeline = go.Figure()
            colors = ['#00b894', '#0984e3', '#fdcb6e', '#1e2a38', '#e63946', '#636e72']

            for i, row in agent_data.iterrows():
                duration = row['Duration']
                if duration > 99:
                    duration_text = f"{duration / 60:.2f}m"
                    hover_duration = f"{duration / 60:.2f} min ({duration:.1f}s)"
                else:
                    duration_text = f"{duration:.1f}s"
                    hover_duration = f"{duration:.1f}s"

                fig_timeline.add_trace(go.Bar(
                    x=[row['Agent']],
                    y=[duration],
                    marker_color=colors[i % len(colors)],
                    name=row['Agent'],
                    text=[duration_text],
                    textposition='inside',
                    hovertemplate=f"<b>{row['Agent']}</b><br>Duration: {hover_duration}<extra></extra>"
                ))

            fig_timeline.update_layout(
                title="Per-Agent Duration",
                xaxis_title="Agent",
                yaxis_title="Duration (seconds)",
                barmode='group',
                height=400,
                showlegend=False,
                hovermode='closest'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)

        with tab2:
            gpu_info = parse_gpu_info()
            model_info = load_model_info_from_json()

            total_gpu_mem = 0.0
            total_disk = 0.0
            for node in gpu_info.get("nodes", []):
                total_gpu_mem += node.get("per_gpu_memory_gb", 0) * max(node.get("allocatable_gpus", 0), 1)
                total_disk += node.get("node_storage_gb", 0)

            total_vram_required = 0.0
            for m in model_info.get("models", []):
                total_vram_required += m.get("estimated_vram_gb", 0)

            if total_gpu_mem == 0:
                total_gpu_mem = get_value("resources.gpu_memory_available_gb", 0)
            if total_vram_required == 0:
                total_vram_required = get_value("resources.gpu_memory_required_gb", 0)
            if total_disk == 0:
                total_disk = get_value("resources.disk_space_gb", 0)

            resources = []
            values = []
            bar_colors = []
            if total_gpu_mem > 0:
                resources.append("GPU Memory (Available)")
                values.append(round(total_gpu_mem, 2))
                bar_colors.append("#00b894")
            if total_vram_required > 0:
                resources.append("GPU Memory (Required)")
                values.append(round(total_vram_required, 2))
                bar_colors.append("#fdcb6e")
            if total_disk > 0:
                resources.append("Node Storage (Total)")
                values.append(round(total_disk, 2))
                bar_colors.append("#0984e3")

            if resources:
                fig_resources = go.Figure()
                fig_resources.add_trace(go.Bar(
                    x=resources,
                    y=values,
                    marker_color=bar_colors,
                    text=[f"{v} GB" for v in values],
                    textposition='outside',
                    hovertemplate="<b>%{x}</b><br>%{y} GB<extra></extra>"
                ))
                fig_resources.update_layout(
                    title="Resource Usage Overview",
                    xaxis_title="Resource Type",
                    yaxis_title="Value (GB)",
                    height=400,
                    showlegend=False,
                )
                st.plotly_chart(fig_resources, use_container_width=True)
            else:
                st.info("No resource data available yet.")
        
        st.session_state.workflow_completed = True
        st.markdown("---")
    
    # Auto-advance workflow steps and run agent command
    if not st.session_state.workflow_completed and st.session_state.workflow_step < 6:
        # Run the actual agent command when workflow starts
        if st.session_state.workflow_step == 1 and not st.session_state.agent_command_output:
            # Set agent start time for timeline tracking
            if st.session_state.agent_start_time is None:
                st.session_state.agent_start_time = time.time()
            if st.session_state.get("agent_run_info_dir") is None:
                _ensure_run_info_dir()

            placeholder = st.empty()
            with placeholder:
                with st.spinner("Running supervisor agent..."):
                    try:
                        # Run the CLI command: agent --config <config_path>
                        config_path = st.session_state.config_path

                        # Find the agent command - try venv first, then system PATH
                        project_dir = os.path.dirname(os.path.abspath(__file__))
                        venv_agent = os.path.join(project_dir, ".venv", "bin", "agent")
                        cmd = None
                        
                        if os.path.exists(venv_agent):
                            cmd = [venv_agent, "--config", config_path]
                        else:
                            import shutil as _shutil
                            agent_path = _shutil.which("agent")
                            if agent_path:
                                cmd = [agent_path, "--config", config_path]
                            else:
                                python_cmd = _shutil.which("python3") or _shutil.which("python")
                                if python_cmd:
                                    cmd = [python_cmd, "-m", "runtimes_dep_agent.execute_agent", "--config", config_path]
                                else:
                                    raise FileNotFoundError("Could not find 'agent' command or Python. Make sure the package is installed.")
                        
                        # Ensure venv's bin is in PATH
                        env = os.environ.copy()
                        venv_bin = os.path.join(project_dir, ".venv", "bin")
                        if os.path.exists(venv_bin):
                            env["PATH"] = f"{venv_bin}:{env.get('PATH', '')}"

                        if st.session_state.oc_login_command:
                            env["OC_LOGIN_COMMAND"] = st.session_state.oc_login_command
                        if st.session_state.vllm_runtime_image:
                            env["VLLM_RUNTIME_IMAGE"] = st.session_state.vllm_runtime_image
                        if st.session_state.registry_host:
                            env["REGISTRY_HOST"] = st.session_state.registry_host
                        # Persist artifacts under <repo>/info so the UI, subprocess, and editor see the same files.
                        env["AGENT_RUN_INFO_DIR"] = str(_get_info_dir())
                        if st.session_state.get("qa_skip_namespace_delete"):
                            env["QA_SKIP_NAMESPACE_DELETE"] = "1"
                        else:
                            env.pop("QA_SKIP_NAMESPACE_DELETE", None)

                        if live_output_placeholder is None:
                            live_output_placeholder = st.empty()

                        env["PYTHONUNBUFFERED"] = "1"
                        returncode, output = stream_agent_command(
                            cmd,
                            env=env,
                            cwd=project_dir,
                            live_output_placeholder=live_output_placeholder,
                            timeout_sec=2100,
                            tail_lines=200,
                        )
                        if returncode != 0:
                            output = f"Command exited with code {returncode}\n\n{output}"
                        
                        st.session_state.agent_command_output = output
                        st.session_state.agent_output_text = output
                        
                        # Track supervisor completion time (when agent_output_text is available)
                        if st.session_state.agent_timestamps["supervisor"] is None:
                            st.session_state.agent_timestamps["supervisor"] = time.time()
                        
                        # Advance to next step
                        st.session_state.workflow_step = 2
                    except subprocess.TimeoutExpired as exc:
                        timeout_message = "Error: Command timed out after 35 minutes"
                        partial_output = exc.output or ""
                        if partial_output:
                            timeout_message = f"{timeout_message}\n\n{partial_output}"
                        st.session_state.agent_command_output = timeout_message
                        st.session_state.agent_output_text = timeout_message
                        st.session_state.workflow_step = 6
                    except Exception as e:
                        st.session_state.agent_command_output = f"Error running agent command: {str(e)}"
                        st.session_state.agent_output_text = f"Error running agent command: {str(e)}"
                        st.session_state.workflow_step = 6
            st.rerun()
        elif st.session_state.workflow_step >= 2 and st.session_state.workflow_step < 6:
            # Simulate progress through remaining workflow steps (no spinner needed, already shown above)
            time.sleep(1.5)  # Shorter delay since agent already ran
            st.session_state.workflow_step += 1
            st.rerun()

# Display Full Agent Output at the bottom after all charts
if st.session_state.agent_output_text:
    st.subheader("Full Agent Output")
    with st.expander("📋 Full Agent Output", expanded=False):
        st.code(_strip_ansi(st.session_state.agent_output_text))
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #636e72; font-size: 0.78rem; padding: 24px 0 8px 0; font-family: 'Red Hat Text', sans-serif;">
        Model Runtimes Agent &bull; Red Hat OpenShift AI &bull; Developed by
        <a href="https://github.com/Raghul-M" target="_blank" style="color: #0984e3; text-decoration: none; font-weight: 600;">Raghul M</a>
    </div>
    """,
    unsafe_allow_html=True,
)
