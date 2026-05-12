"""QA Specialist: KServe InferenceService deployment validation (sequential, self-healing)."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Callable

import yaml
from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from ...qa_kserve.pipeline import run_kserve_deployment_qa
from ...utils.path_utils import detect_repo_root
from . import SpecialistSpec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _infer_registry_from_modelcar(modelcar_cfg: dict) -> str | None:
    """Infer a single registry host from model-car image fields, if possible."""

    def _extract_registry_host(image: str) -> str | None:
        if not image:
            return None
        trimmed = image.strip()
        for prefix in ("oci://", "docker://", "http://", "https://"):
            if trimmed.startswith(prefix):
                trimmed = trimmed[len(prefix) :]
                break
        if not trimmed:
            return None
        parts = trimmed.split("/", 1)
        host = parts[0].strip()
        return host or None

    if not isinstance(modelcar_cfg, dict):
        return None
    model_block = modelcar_cfg.get("model-car")
    if isinstance(model_block, dict):
        model_entries = [model_block]
    elif isinstance(model_block, list):
        model_entries = model_block
    else:
        return None

    registries: set[str] = set()
    for entry in model_entries:
        if not isinstance(entry, dict):
            continue
        image = entry.get("image")
        if not isinstance(image, str):
            continue
        host = _extract_registry_host(image)
        if host:
            registries.add(host)

    if len(registries) == 1:
        return next(iter(registries))
    return None


def _resolve_registry_host(repo_root: Path) -> str | None:
    explicit = os.environ.get("REGISTRY_HOST", "").strip()
    if explicit:
        return explicit
    gen = repo_root / "config-yaml" / "sample_modelcar_config.generated.yaml"
    base = repo_root / "config-yaml" / "sample_modelcar_config.base.yaml"
    path = gen if gen.exists() else base
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        return _infer_registry_from_modelcar(cfg or {})
    except OSError:
        return None


def build_qa_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
    info_dir: Path | None = None,
) -> SpecialistSpec:
    """Return the QA specialist agent and the supervisor-facing tool."""
    effective_info_dir = info_dir
    _repo = detect_repo_root()

    agents_md = _repo / "deployment-yamls" / "agents.md"
    playbook_hint = ""
    if agents_md.exists():
        try:
            text = agents_md.read_text(encoding="utf-8")
            playbook_hint = (
                "\n\nOperational playbook (excerpt from deployment-yamls/agents.md):\n"
                + text[:6000]
                + ("\n..." if len(text) > 6000 else "")
            )
        except OSError:
            pass

    @tool
    def run_kserve_deployment_qa_tool(runtime_image: str, gpu_provider: str) -> str:
        """
        Deploy deployable models to namespace model-validation using deployment-yamls templates:
        apply OCI pull secret, then InferenceServices in ascending model image size order.
        Monitors readiness, events, and container logs; on failure proposes remediation (serving args +
        CPU/memory/GPU) via the specialist LLM when available, otherwise uses bounded heuristic retries.

        Requires environment: KUBECONFIG, OCI_REGISTRY_PULL_SECRET (base64 .dockerconfigjson or raw JSON),
        REGISTRY_HOST (or a single registry inferable from model-car). Optional: VLLM_RUNTIME_IMAGE,
        KSERVE_SERVING_RUNTIME_NAME, KSERVE_MODEL_FORMAT, QA_PER_MODEL_TIMEOUT_S, QA_MAX_GPU_COUNT,
        QA_SKIP_SERVING_RUNTIME_APPLY (set to 1 to skip applying serving-runtime.yaml.template when the runtime already exists).
        Post-deploy: OpenAI-style smoke POST to /v1/chat/completions (skip with QA_SKIP_POST_DEPLOY_SMOKE=1), scale-to-zero (skip with QA_SKIP_SCALE_TO_ZERO=1), delete namespace on full success (skip with QA_SKIP_NAMESPACE_DELETE=1). Optional QA_SMOKE_MODEL_ID, QA_SMOKE_USER_MESSAGE, QA_SMOKE_MAX_TOKENS, QA_SMOKE_TIMEOUT_S, QA_SMOKE_TLS_VERIFY.

        :param runtime_image: vLLM / runtime image used for annotations and validation (see accelerator JSON).
        :param gpu_provider: e.g. NVIDIA, AMD — affects GPU resource requests.
        """
        repo_root = detect_repo_root()
        reg = _resolve_registry_host(repo_root)
        oci = os.environ.get("OCI_REGISTRY_PULL_SECRET", "").strip()

        if not reg:
            msg = (
                "QA_ERROR:REGISTRY_HOST_MISSING Set REGISTRY_HOST or ensure model-car lists "
                "a single registry host."
            )
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg
        if not oci:
            msg = "QA_ERROR:OCI_PULL_SECRET_MISSING Set OCI_REGISTRY_PULL_SECRET (base64 .dockerconfigjson)."
            logger.error(msg)
            print(f"[QA] {msg}", flush=True)
            return msg

        return run_kserve_deployment_qa(
            runtime_image=runtime_image,
            gpu_provider=gpu_provider,
            registry_host=reg,
            oci_pull_secret=oci,
            precomputed_requirements=precomputed_requirements,
            info_dir=effective_info_dir,
            repo_root=repo_root,
            llm=llm,
        )

    prompt = (
        "You are a QA Specialist responsible for validating ML model deployments on OpenShift / Kubernetes "
        "using KServe InferenceServices.\n\n"
        "You have a tool `run_kserve_deployment_qa_tool` that applies manifests under deployment-yamls/, "
        "creates namespace model-validation if needed, applies the registry pull secret, deploys each "
        "deployable model from deployment_matrix.json + generated model-car YAML in ascending container "
        "image size order, waits for Ready, tails storage-initializer and kserve-container logs when useful, "
        "and retries with LLM-proposed args/resources (or heuristic memory / max-model-len tuning) on "
        "recoverable failures.\n\n"
        "When invoked by the supervisor:\n"
        "1. Call `run_kserve_deployment_qa_tool` with the runtime_image from the supervisor (accelerator vLLM image) "
        "and gpu_provider.\n"
        "2. Inspect the returned string: it starts with QA_OK: or QA_ERROR:.\n"
        "3. Summarize pass/fail per model and overall, without asking for kubeconfig or secret contents.\n"
        + playbook_hint
    )

    agent = create_agent(
        llm,
        tools=[run_kserve_deployment_qa_tool],
        system_prompt=prompt,
    )

    @tool
    def analyze_qa_results(request: str, runtime_image: str, gpu_provider: str):
        """
        Supervisor-facing entrypoint. Pass:
            - request: e.g. \"Run QA and summarize validation results.\"
            - runtime_image: vLLM runtime image (from accelerator JSON / VLLM_RUNTIME_IMAGE).
            - gpu_provider: e.g. NVIDIA or AMD.
        """
        qa_input = (
            f"{request}\n\n"
            f"RUNTIME_IMAGE::{runtime_image}\n"
            f"GPU_PROVIDER::{gpu_provider}\n"
            "You MUST call `run_kserve_deployment_qa_tool` with this runtime_image."
        )

        result = agent.invoke({"messages": [{"role": "user", "content": qa_input}]})
        return extract_text(result)

    analyze_qa_results.name = "analyze_qa_results"

    return SpecialistSpec(
        name="qa_specialist",
        agent=agent,
        tool=analyze_qa_results,
    )


__all__ = ["build_qa_specialist"]
