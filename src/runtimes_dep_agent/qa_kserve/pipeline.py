"""Sequential KServe deploy QA: apply manifests, watch, heal."""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import yaml

from ..utils.path_utils import detect_repo_root
from .heuristics import classify_pod_json, logs_hint_oom
from .oc_cli import run_oc
from .render import (
    build_registry_secret_yaml,
    halve_max_model_len_args,
    load_inference_template,
    normalize_dockerconfig_b64,
    pick_cpu_memory,
    render_inference_service,
    sanitize_k8s_name,
    validate_yaml_document,
)

logger = logging.getLogger(__name__)

QA_NAMESPACE = "model-validation"


def _kubeconfig_path() -> Path:
    return Path(os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config")))


def _append_report(parts: list[str], msg: str) -> None:
    parts.append(msg)
    print(f"[QA] {msg}", flush=True)


def _ensure_namespace(namespace: str, log: list[str]) -> bool:
    r = run_oc(["get", "namespace", namespace], timeout=30)
    if r.returncode == 0:
        return True
    r2 = run_oc(["create", "namespace", namespace], timeout=60)
    if r2.returncode != 0:
        _append_report(log, f"QA_ERROR:NAMESPACE_FAILED {r2.stderr or r2.stdout}")
        return False
    _append_report(log, f"created namespace {namespace}")
    return True


def _apply_yaml_document(doc_yaml: str, log: list[str], *, timeout: float = 120) -> bool:
    try:
        validate_yaml_document(doc_yaml)
    except ValueError as e:
        _append_report(log, f"QA_ERROR:YAML_INVALID {e}")
        return False

    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        delete=False,
        encoding="utf-8",
    ) as tmp:
        tmp.write(doc_yaml)
        path = tmp.name

    try:
        r = run_oc(["apply", "-f", path], timeout=timeout)
        if r.returncode != 0:
            _append_report(log, f"QA_ERROR:APPLY_FAILED {r.stderr or r.stdout}")
            return False
        return True
    finally:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass


def _delete_isvc(name: str, log: list[str]) -> None:
    r = run_oc(
        ["delete", "inferenceservice", name, "-n", QA_NAMESPACE, "--ignore-not-found=true"],
        timeout=180,
    )
    if r.returncode != 0:
        _append_report(log, f"warn delete isvc: {r.stderr or r.stdout}")
    else:
        # Wait for resource removal
        time.sleep(3)


def _fetch_pod_logs(isvc_name: str, container_hint: str, log: list[str]) -> str:
    """Best-effort logs from first matching pod."""
    r = run_oc(
        [
            "get",
            "pods",
            "-n",
            QA_NAMESPACE,
            "-l",
            f"serving.kserve.io/inferenceservice={isvc_name}",
            "-o",
            "json",
        ],
        timeout=60,
    )
    if r.returncode != 0 or not (r.stdout or "").strip():
        return ""
    try:
        doc = json.loads(r.stdout)
        items = doc.get("items") or []
        if not items:
            return ""
        pod_obj = items[0]
        pod_name = pod_obj.get("metadata", {}).get("name") or ""
    except json.JSONDecodeError:
        return ""
    if not pod_name:
        return ""

    skip_containers = {"pauser", "queue-proxy"}

    containers = []
    for c in pod_obj.get("spec", {}).get("containers") or []:
        if isinstance(c, dict) and c.get("name"):
            containers.append(c["name"])
    target = None
    for c in containers:
        if container_hint in c or c in {"kserve-container", "storage-initializer"}:
            target = c
            break
    if target is None:
        for c in containers:
            if c not in skip_containers:
                target = c
                break
    if target is None:
        return ""

    r2 = run_oc(
        ["logs", pod_name, "-n", QA_NAMESPACE, "-c", target, "--tail=120"],
        timeout=60,
    )
    if r2.returncode != 0:
        _append_report(log, f"(logs {target}) {r2.stderr or ''}")
        return ""
    return r2.stdout or ""


def _wait_ready_or_failure(
    isvc_name: str,
    *,
    deadline_s: float,
    poll_s: float,
    log: list[str],
) -> tuple[bool, str]:
    """Wait until Ready=True or detect failure / timeout."""
    deadline = time.monotonic() + deadline_s
    last_diag = ""

    while time.monotonic() < deadline:
        r = run_oc(
            ["get", "inferenceservice", isvc_name, "-n", QA_NAMESPACE, "-o", "json"],
            timeout=60,
        )
        if r.returncode == 0 and r.stdout:
            try:
                doc = json.loads(r.stdout)
                for cond in doc.get("status", {}).get("conditions") or []:
                    if cond.get("type") == "Ready" and cond.get("status") == "True":
                        return True, "Ready"
                    if cond.get("type") == "Ready" and cond.get("status") == "False":
                        msg = cond.get("message") or cond.get("reason") or ""
                        last_diag = msg[:500]
            except json.JSONDecodeError:
                pass

        rp = run_oc(
            [
                "get",
                "pods",
                "-n",
                QA_NAMESPACE,
                "-l",
                f"serving.kserve.io/inferenceservice={isvc_name}",
                "-o",
                "json",
            ],
            timeout=60,
        )
        if rp.returncode == 0 and rp.stdout:
            kind, detail = classify_pod_json(rp.stdout)
            if kind in ("oom", "image_pull", "crashloop"):
                return False, f"{kind}:{detail}"

        time.sleep(poll_s)

    return False, f"timeout:{last_diag}"


def _load_deployment_matrix(matrix_path: Path) -> list[dict]:
    if not matrix_path.exists():
        return []
    with open(matrix_path, encoding="utf-8") as f:
        data = json.load(f)
    return data if isinstance(data, list) else []


def _load_generated_modelcar(repo_root: Path) -> dict:
    gen = repo_root / "config-yaml" / "sample_modelcar_config.generated.yaml"
    base = repo_root / "config-yaml" / "sample_modelcar_config.base.yaml"
    path = gen if gen.exists() else base
    if not path.exists():
        raise FileNotFoundError(f"No model-car at {gen} or {base}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _model_car_entries(cfg: dict) -> list[dict]:
    block = cfg.get("model-car")
    if isinstance(block, dict):
        return [block]
    if isinstance(block, list):
        return [m for m in block if isinstance(m, dict)]
    return []


def _gpu_count_from_entry(entry: dict) -> int:
    sa = entry.get("serving_arguments") or {}
    if isinstance(sa, dict):
        g = sa.get("gpu_count")
        if isinstance(g, int) and g >= 0:
            return g
        try:
            return max(0, int(g))
        except (TypeError, ValueError):
            pass
    return 1


def run_kserve_deployment_qa(
    *,
    runtime_image: str,
    gpu_provider: str,
    registry_host: str | None = None,
    oci_pull_secret: str | None = None,
    precomputed_requirements: dict | None = None,
    info_dir: Path | None = None,
    repo_root: Path | None = None,
    per_model_timeout_s: int | None = None,
    max_heal_retries: int = 3,
    poll_interval_s: float = 12.0,
) -> str:
    """
    Deploy deployable models sequentially (small image first), validate InferenceServices,
    apply bounded healing on OOM-style failures.

    Returns a string starting with QA_OK: or QA_ERROR: for downstream parsers.
    """
    log: list[str] = []
    root = repo_root or detect_repo_root()
    eff_registry = (registry_host or os.environ.get("REGISTRY_HOST", "")).strip()
    eff_secret = (oci_pull_secret or os.environ.get("OCI_REGISTRY_PULL_SECRET", "")).strip()
    eff_runtime = (runtime_image or "").strip() or os.environ.get("VLLM_RUNTIME_IMAGE", "").strip()

    kc = _kubeconfig_path()
    if not kc.exists():
        msg = f"QA_ERROR:KUBECONFIG_MISSING {kc}"
        _append_report(log, msg)
        return msg

    if not eff_registry:
        msg = "QA_ERROR:REGISTRY_HOST_MISSING Set REGISTRY_HOST or pass registry_host."
        _append_report(log, msg)
        return msg
    if not eff_secret:
        msg = "QA_ERROR:OCI_PULL_SECRET_MISSING Set OCI_REGISTRY_PULL_SECRET or pass oci_pull_secret."
        _append_report(log, msg)
        return msg
    if not eff_runtime:
        msg = "QA_ERROR:VLLM_RUNTIME_IMAGE_MISSING No vLLM runtime image provided."
        _append_report(log, msg)
        return msg

    matrix_path = (info_dir / "deployment_matrix.json") if info_dir else root / "info" / "deployment_matrix.json"
    matrix = _load_deployment_matrix(matrix_path)
    deployable_names = {
        e["model_name"]
        for e in matrix
        if isinstance(e, dict) and e.get("deployable") is True and e.get("model_name")
    }

    try:
        mc = _load_generated_modelcar(root)
    except FileNotFoundError as e:
        msg = f"QA_ERROR:MODELCAR_NOT_FOUND {e}"
        _append_report(log, msg)
        return msg

    entries = [m for m in _model_car_entries(mc) if m.get("name") in deployable_names]
    if not entries:
        msg = "QA_ERROR:NO_DEPLOYABLE_MODELS Nothing matched deployment_matrix.json + generated model-car."
        _append_report(log, msg)
        return msg

    req_map = precomputed_requirements or {}
    enriched: list[tuple[dict, float]] = []
    for entry in entries:
        name = entry.get("name") or ""
        sz = 0.0
        if name in req_map and isinstance(req_map[name], dict):
            sz = float(req_map[name].get("model_size_gb") or 0)
        enriched.append((entry, sz))

    enriched.sort(key=lambda x: (x[1], x[0].get("name") or ""))

    oci_secret_name = os.environ.get("OCI_REGISTRY_SECRET_NAME", "oci-registry-pull-secret")
    serving_runtime = os.environ.get("KSERVE_SERVING_RUNTIME_NAME", "vllm-runtime")
    model_format = os.environ.get("KSERVE_MODEL_FORMAT", "huggingface")
    timeout_per = per_model_timeout_s or int(os.environ.get("QA_PER_MODEL_TIMEOUT_S", "900"))

    _append_report(log, "Starting KServe deployment QA (sequential, small-to-large image).")

    if not _ensure_namespace(QA_NAMESPACE, log):
        return "\n".join(log)

    docker_b64 = normalize_dockerconfig_b64(eff_secret)
    secret_yaml = build_registry_secret_yaml(
        registry_host=eff_registry,
        dockerconfigjson_b64=docker_b64,
        secret_name=oci_secret_name,
        namespace=QA_NAMESPACE,
    )
    if not _apply_yaml_document(secret_yaml, log):
        return "\n".join(log)

    template_text = load_inference_template(root)

    outcomes: list[str] = []
    for entry, _sz in enriched:
        model_name = entry.get("name") or "unknown"
        model_image = (entry.get("image") or "").strip()
        if not model_image:
            outcomes.append(f"{model_name}:skipped_no_image")
            continue

        args = []
        sa = entry.get("serving_arguments") or {}
        if isinstance(sa, dict):
            args = list(sa.get("args") or [])

        isvc_name = sanitize_k8s_name(str(model_name))
        gpu_n = _gpu_count_from_entry(entry)
        if gpu_provider.upper() in ("CPU", "NONE", ""):
            gpu_n = 0

        req_info = req_map.get(model_name) if isinstance(req_map.get(model_name), dict) else {}
        vram = None
        if req_info:
            vram = req_info.get("required_vram_gb")
            try:
                vram = float(vram) if vram is not None else None
            except (TypeError, ValueError):
                vram = None

        ok_model = False
        failure_reason = ""
        cur_args = list(args)
        mem_bump = 0

        for attempt in range(max_heal_retries + 1):
            cpu_req, mem_req, cpu_lim, mem_lim = pick_cpu_memory(
                required_vram_gb=vram,
                heal_bump=mem_bump,
            )

            body = render_inference_service(
                template_text=template_text,
                isvc_name=isvc_name,
                model_image=model_image,
                vllm_runtime_image=eff_runtime,
                serving_runtime_name=serving_runtime,
                model_format=model_format,
                args=cur_args,
                oci_secret_name=oci_secret_name,
                cpu_request=cpu_req,
                memory_request=mem_req,
                cpu_limit=cpu_lim,
                memory_limit=mem_lim,
                gpu_count=gpu_n,
            )

            try:
                validate_yaml_document(body)
            except ValueError as e:
                outcomes.append(f"{model_name}:QA_ERROR:YAML_INVALID:{e}")
                failure_reason = str(e)
                break

            _delete_isvc(isvc_name, log)
            if not _apply_yaml_document(body, log, timeout=180):
                outcomes.append(f"{model_name}:apply_failed")
                failure_reason = "apply_failed"
                break

            ready, detail = _wait_ready_or_failure(
                isvc_name,
                deadline_s=float(timeout_per),
                poll_s=poll_interval_s,
                log=log,
            )

            if ready:
                outcomes.append(f"{model_name}:OK")
                ok_model = True
                _append_report(log, f"{model_name} Ready.")
                break

            _append_report(log, f"{model_name} not ready: {detail}")

            rp = run_oc(
                [
                    "get",
                    "pods",
                    "-n",
                    QA_NAMESPACE,
                    "-l",
                    f"serving.kserve.io/inferenceservice={isvc_name}",
                    "-o",
                    "json",
                ],
                timeout=60,
            )
            kind = "unknown"
            if rp.returncode == 0 and rp.stdout:
                kind, _kdetail = classify_pod_json(rp.stdout)

            log_snip = ""
            for hint in ("storage-initializer", "kserve-container"):
                chunk = _fetch_pod_logs(isvc_name, hint, log)
                if chunk:
                    log_snip += chunk + "\n"
            if logs_hint_oom(log_snip):
                kind = "oom"

            if kind == "image_pull":
                outcomes.append(f"{model_name}:QA_ERROR:IMAGE_PULL")
                break

            if attempt < max_heal_retries:
                mem_bump += 1
                cur_args = halve_max_model_len_args(cur_args)
                _append_report(
                    log,
                    f"Heal attempt {attempt + 1}/{max_heal_retries} after {detail} (memory bump, args adjust).",
                )
                continue

            failure_reason = detail or kind
            outcomes.append(f"{model_name}:FAIL:{failure_reason}")
            break

    bad = [x for x in outcomes if "QA_ERROR" in x or ":FAIL:" in x]
    summary = "; ".join(outcomes)
    if bad:
        return "QA_ERROR:KSERVE_DEPLOYMENT_FAILED " + summary + "\n" + "\n".join(log)

    return "QA_OK:" + summary + "\n" + "\n".join(log)
