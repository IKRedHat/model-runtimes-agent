"""Main entry point for running the supervisor agent."""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from langgraph.errors import GraphRecursionError
from pathlib import Path

# Suppress noisy loggers before any library imports
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("runtimes_dep_agent").setLevel(logging.WARNING)

from .utils.path_utils import detect_repo_root
from .preflight import (
    run_preflight_checks,
    preflight_ok,
    ProgressLogger,
    BOLD,
    CYAN,
    GREEN,
    RED,
    YELLOW,
    DIM,
    CHECK,
    CROSS,
    RESET,
)
from runtimes_dep_agent.agent.llm_agent import LLMAgent


DEFAULT_CONFIG_PATH = "config-yaml/sample_modelcar_config.yaml"

SUPERVISOR_TRIGGER_MESSAGE = (
    "Start supervisor agent operation. Receive model-car configuration "
    "report from config specialist and make deployment decisions."
)

# Map tool function names to human-friendly labels
_TOOL_LABELS = {
    "analyze_model_config": "Configuration Specialist",
    "analyze_accelerator": "Accelerator Specialist",
    "get_detailed_gpu_information": "Accelerator Specialist (GPU details)",
    "get_accelerator_metadata_json": "Accelerator Specialist (metadata)",
    "analyze_deployment_decision": "Decision Specialist",
    "analyze_qa_results": "QA Specialist",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the supervisor agent end-to-end.",
    )
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_PATH,
        help="Path to the model-car YAML config file to preload.",
    )
    parser.add_argument(
        "--model",
        default="gemini-2.5-pro",
        help="Model name to use for the supervisor agent.",
    )
    parser.add_argument(
        "--gemini-api-key",
        default=None,
        help="Gemini API key (falls back to GEMINI_API_KEY env var).",
    )
    parser.add_argument(
        "--oci-pull-secret",
        default=None,
        help="OCI registry pull secret (falls back to OCI_REGISTRY_PULL_SECRET env var).",
    )
    parser.add_argument(
        "--vllm-runtime-image",
        default=None,
        help="vLLM runtime image override (falls back to VLLM_RUNTIME_IMAGE env var).",
    )
    parser.add_argument(
        "--registry-host",
        default=None,
        help="OCI registry host for QA deploy (falls back to REGISTRY_HOST env var).",
    )
    parser.add_argument(
        "--oc-login",
        default=None,
        help=(
            'Full oc login command to execute before running the agent. '
            'Example: --oc-login "oc login --token=sha256~... --server=https://..."'
        ),
    )
    parser.add_argument(
        "--report-output",
        default="report.html",
        help="Path for the generated HTML report (default: report.html).",
    )
    return parser.parse_args()


def _apply_env_overrides(args: argparse.Namespace) -> str:
    """Set environment variables from CLI flags and return the resolved API key."""
    if args.gemini_api_key:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key
    if args.oci_pull_secret:
        os.environ["OCI_REGISTRY_PULL_SECRET"] = args.oci_pull_secret
    if args.vllm_runtime_image:
        os.environ["VLLM_RUNTIME_IMAGE"] = args.vllm_runtime_image
    if args.registry_host:
        os.environ["REGISTRY_HOST"] = args.registry_host

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            f"{RED}{CROSS} GEMINI_API_KEY is required. "
            f"Pass --gemini-api-key or set the environment variable.{RESET}"
        )
        sys.exit(1)
    return api_key


def _run_oc_login(command: str, progress: ProgressLogger) -> None:
    """Execute the user-supplied oc login command string."""
    progress.section("OpenShift Login")
    tokens = shlex.split(command)
    if tokens and tokens[0] != "oc":
        tokens = ["oc"] + tokens

    # Only allow 'oc login'; reject any other oc subcommand
    subcommand = None
    for t in tokens[1:]:
        if not t.startswith("-"):
            subcommand = t
            break
    if subcommand != "login":
        progress.fail("Only 'oc login' is allowed; other oc subcommands are not permitted.")
        sys.exit(1)

    server = next((t.split("=", 1)[1] for t in tokens if t.startswith("--server=")), None)
    if not server:
        for i, t in enumerate(tokens):
            if t == "--server" and i + 1 < len(tokens):
                server = tokens[i + 1]
                break
    target = server or "cluster"
    progress.detail(f"Logging in to {target} ...")

    try:
        result = subprocess.run(
            tokens,
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0:
            progress.success("Logged in")
        else:
            stderr = (result.stderr or result.stdout or "").strip()
            progress.fail(f"Login failed: {stderr}")
            sys.exit(1)
    except FileNotFoundError:
        progress.fail("oc command not found")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        progress.fail("oc login timed out")
        sys.exit(1)


def _extract_message_text(message) -> str:
    """Pull plain text from a LangChain message object."""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
        return "\n".join(p for p in parts if p)
    return str(content)


def _summarize_tool_response(tool_name: str, text: str) -> str:
    """Extract a clean one-line summary from a tool response."""
    if not text:
        return ""

    lines = [l.strip() for l in text.splitlines() if l.strip()]
    # Skip checklist lines like "[x] ..."
    content_lines = [l for l in lines if not l.startswith("[x]") and not l.startswith("[ ]")]
    if not content_lines:
        content_lines = lines

    first = content_lines[0] if content_lines else ""
    # Strip markdown prefixes
    first = first.lstrip("#").lstrip("-").lstrip("*").strip()
    if len(first) > 120:
        first = first[:117] + "..."
    return first


def _run_streaming(agent: LLMAgent, progress: ProgressLogger) -> str:
    """Stream the supervisor pipeline, printing live output to the terminal."""
    progress.section("Running Supervisor Pipeline")
    final_text = ""
    seen_tools: set[str] = set()
    call_count = 0

    try:
        for event in agent.stream_supervisor(SUPERVISOR_TRIGGER_MESSAGE):
            for node_name, node_output in event.items():
                messages = node_output.get("messages", [])
                for msg in messages:
                    msg_type = getattr(msg, "type", None) or type(msg).__name__

                    if msg_type == "ai":
                        tool_calls = getattr(msg, "tool_calls", None) or []
                        for tc in tool_calls:
                            tool_name = tc.get("name", "") if isinstance(tc, dict) else getattr(tc, "name", "")
                            label = _TOOL_LABELS.get(tool_name, tool_name)
                            if tool_name not in seen_tools:
                                call_count += 1
                                seen_tools.add(tool_name)
                                progress.step(f"{label}")
                            else:
                                progress.detail(f"{DIM}↳ {label} (follow-up){RESET}")
                            sys.stdout.flush()

                        text = _extract_message_text(msg)
                        if text and not tool_calls:
                            final_text = text

                    elif msg_type == "tool":
                        tool_name = getattr(msg, "name", "")
                        label = _TOOL_LABELS.get(tool_name, tool_name)
                        text = _extract_message_text(msg)
                        summary = _summarize_tool_response(tool_name, text)
                        content = f"{(summary or '')}\n{(text or '')}".lower()
                        is_error = any(
                            ind in content
                            for ind in ("unauthoriz", "failed", "error")
                        )
                        if is_error:
                            progress.fail(f"{label} failed")
                            if summary:
                                progress.detail(f"{DIM}{summary}{RESET}")
                        elif summary:
                            progress.success(f"{label} done")
                            progress.detail(f"{DIM}{summary}{RESET}")
                        else:
                            progress.success(f"{label} done")
                        sys.stdout.flush()

    except GraphRecursionError:
        final_text = "Error: maximum recursion depth reached."
        progress.fail(final_text)

    return final_text


def main() -> None:
    start_time = time.time()
    args = _parse_args()

    # 1. Pre-flight checks
    results = run_preflight_checks()
    if not preflight_ok(results):
        print(f"\n{RED}Aborting: required tools are missing.{RESET}")
        sys.exit(1)

    progress = ProgressLogger(total_steps=5)

    # Per-run artifact directory (use app-provided path when running under Streamlit UI)
    env_info_dir = os.environ.get("AGENT_RUN_INFO_DIR")
    if env_info_dir:
        info_dir = Path(env_info_dir)
        info_dir.mkdir(parents=True, exist_ok=True)
        run_dir = None  # app owns it; no cleanup
    else:
        run_dir = Path(tempfile.mkdtemp(prefix="agent_run_"))
        info_dir = run_dir / "info"
        info_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 2. oc login
        oc_login_cmd = args.oc_login or os.environ.get("OC_LOGIN_COMMAND")
        if oc_login_cmd:
            _run_oc_login(oc_login_cmd, progress)

        # 3. Set env vars from CLI flags
        api_key = _apply_env_overrides(args)

        # 4. Build the agent
        progress.section("Initializing Agent")
        progress.step("Loading configuration & building specialists")
        agent = LLMAgent(
            api_key=api_key,
            model=args.model,
            bootstrap_config=args.config,
            info_dir=info_dir,
        )
        progress.success("Agent ready")

        # 5. Stream the supervisor pipeline with live output
        output_text = _run_streaming(agent, progress)

        # Write summary to per-run info dir
        summary_path = info_dir / "supervisor_summary.txt"
        with open(summary_path, "w") as f:
            f.write(output_text)

        # Print final report
        print(f"\n{'=' * 60}")
        print(f"{BOLD}  SUPERVISOR REPORT{RESET}")
        print(f"{'=' * 60}\n")
        print(output_text)
        print(f"\n{'=' * 60}")

        # 6. Always generate report
        progress.step("Generating HTML report")
        try:
            from .report.html_report import generate_html_report

            report_path = generate_html_report(
                info_dir=info_dir,
                output_path=Path(args.report_output),
                agent_output=output_text,
                preflight_results=[r.to_dict() for r in results],
            )
            progress.success(f"Report saved to {report_path}")
        except Exception as exc:
            progress.fail(f"Report generation failed: {exc}")

        elapsed = time.time() - start_time
        progress.done(elapsed)
    finally:
        if run_dir is not None:
            try:
                shutil.rmtree(run_dir)
            except OSError:
                pass


if __name__ == "__main__":
    main()
