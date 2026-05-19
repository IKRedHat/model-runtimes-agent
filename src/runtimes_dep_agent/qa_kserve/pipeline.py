"""Sequential KServe deploy QA: apply manifests, watch, LLM-driven heal."""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from ..utils.path_utils import detect_repo_root
from ..validators.accelerator_validator import (
    get_vllm_runtime_image_from_template,
    normalize_gpu_provider_for_vllm_template,
)
from .heuristics import classify_pod_json, logs_hint_oom
from .oc_cli import run_oc
from .post_deploy import (
    delete_namespace,
    patch_isvc_scale_to_zero,
    post_chat_completions_smoke,
    resolve_inference_base_url,
)
from .remediation_llm import propose_remediation
from .render import (
    build_registry_secret_yaml,
    halve_max_model_len_args,
    load_inference_template,
    load_serving_runtime_template,
    normalize_dockerconfig_b64,
    pick_cpu_memory,
    render_inference_service,
    render_serving_runtime,
    sanitize_k8s_name,
    validate_yaml_document,
)

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

QA_NAMESPACE = "model-validation"


def _env_truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "y")


def _kubeconfig_path() -> Path:
    return Path(os.environ.get("KUBECONFIG", os.path.expanduser("~/.kube/config")))


def _append_report(parts: list[str], msg: str) -> None:
    parts.append(msg)
    print(f"[QA] {msg}", flush=True)


def _return_last_qa_error(log: list[str]) -> str:
    """Latest ``QA_ERROR:...`` line so early returns match the QA_OK/QA_ERROR contract."""
    for i in range(len(log) - 1, -1, -1):
        line = log[i].strip()
        if line.startswith("QA_ERROR:"):
            return line
    return "QA_ERROR:UNKNOWN Check [QA] log output above."


def _qa_progress(event: str, *fields: str) -> None:
    """
    Emit stable, parseable progress signals for UI/CLI.
    Format: QA_<EVENT>::field1::field2...
    """
    payload = "::".join([event, *fields])
    print(f"[QA] {payload}", flush=True)


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

    r = run_oc(["apply", "-f", "-"], stdin_input=doc_yaml, timeout=timeout)
    if r.returncode != 0:
        _append_report(log, f"QA_ERROR:APPLY_FAILED {r.stderr or r.stdout}")
        return False
    return True


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


def _fetch_pod_logs(
    isvc_name: str,
    container_hint: str,
    log: list[str],
    *,
    tail_lines: int = 600,
) -> str:
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
        ["logs", pod_name, "-n", QA_NAMESPACE, "-c", target, f"--tail={tail_lines}"],
        timeout=120,
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


def _fetch_recent_events(log: list[str], *, max_lines: int = 80) -> str:
    """Recent namespace events (best-effort tail)."""
    r = run_oc(
        [
            "get",
            "events",
            "-n",
            QA_NAMESPACE,
            "--sort-by=.lastTimestamp",
        ],
        timeout=90,
    )
    if r.returncode != 0:
        _append_report(log, f"(events) {r.stderr or ''}")
        return ""
    lines = (r.stdout or "").splitlines()
    return "\n".join(lines[-max_lines:])


def _pod_json_excerpt(isvc_name: str, log: list[str], *, max_chars: int = 12000) -> str:
    """Truncated pod list JSON for LLM / failure context."""
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
    if r.returncode != 0:
        _append_report(log, f"(pods excerpt) {r.stderr or ''}")
        return ""
    raw = (r.stdout or "").strip()
    return raw[:max_chars]


def _max_gpu_allowed() -> int:
    raw = os.environ.get("QA_MAX_GPU_COUNT", "8").strip()
    try:
        return max(0, int(raw))
    except ValueError:
        return 8


def _image_pull_retries() -> int:
    raw = os.environ.get("QA_IMAGE_PULL_RETRIES", "1").strip()
    try:
        return max(0, min(5, int(raw)))
    except ValueError:
        return 1


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
    max_heal_retries: int = 2,
    poll_interval_s: float = 12.0,
    llm: BaseChatModel | None = None,
) -> str:
    """
    Deploy deployable models sequentially (small image first), validate InferenceServices,
    and apply bounded healing. When ``llm`` is set, remediation uses logs/events/pod JSON to
    propose full replacement serving args and CPU/memory/GPU resources (temperature 0). Otherwise
    a heuristic bump applies (memory tier + halved ``--max-model-len``). At most
    ``max_heal_retries + 1`` deploy attempts per model (default 2 retries → 3 attempts total).
    Image pull failures do not use LLM remediation.

    After each successful Ready: resolve HTTP(S) base URL, POST ``/v1/chat/completions`` (unless ``QA_SKIP_POST_DEPLOY_SMOKE``), patch scale-to-zero (unless ``QA_SKIP_SCALE_TO_ZERO``). If all models succeed, delete ``model-validation`` unless ``QA_SKIP_NAMESPACE_DELETE``.

    **vLLM image:** ``runtime_image`` argument, else ``VLLM_RUNTIME_IMAGE`` env; if both unset, the image is read
    from the cluster RHOAI template (``vllm-cuda-runtime-template`` for ``NONE``/``CPU``/unknown provider,
    or the provider-specific template for NVIDIA/AMD/Spyre/Intel).

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
        tpl_key = normalize_gpu_provider_for_vllm_template(gpu_provider)
        try:
            eff_runtime = get_vllm_runtime_image_from_template(tpl_key)
            _append_report(
                log,
                f"vLLM runtime image from cluster template ({tpl_key}): {eff_runtime}",
            )
        except RuntimeError as e:
            msg = (
                "QA_ERROR:VLLM_RUNTIME_IMAGE_MISSING No vLLM runtime image from tool/env and "
                f"cluster template lookup failed: {e}"
            )
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
        return _return_last_qa_error(log)

    docker_b64 = normalize_dockerconfig_b64(eff_secret)
    secret_yaml = build_registry_secret_yaml(
        registry_host=eff_registry,
        dockerconfigjson_b64=docker_b64,
        secret_name=oci_secret_name,
        namespace=QA_NAMESPACE,
    )
    if not _apply_yaml_document(secret_yaml, log):
        return _return_last_qa_error(log)

    skip_sr = os.environ.get("QA_SKIP_SERVING_RUNTIME_APPLY", "").strip().lower() in (
        "1",
        "true",
        "yes",
    )
    if not skip_sr:
        try:
            sr_template = load_serving_runtime_template(root)
        except FileNotFoundError as e:
            msg = f"QA_ERROR:SERVING_RUNTIME_TEMPLATE_MISSING {e}"
            _append_report(log, msg)
            return msg
        sr_body = render_serving_runtime(
            template_text=sr_template,
            namespace=QA_NAMESPACE,
            serving_runtime_name=serving_runtime,
            vllm_runtime_image=eff_runtime,
            model_format_name=model_format,
        )
        try:
            validate_yaml_document(sr_body)
        except ValueError as e:
            msg = f"QA_ERROR:SERVING_RUNTIME_YAML_INVALID {e}"
            _append_report(log, msg)
            return msg
        _append_report(log, f"Applying ServingRuntime {serving_runtime} in {QA_NAMESPACE}.")
        _qa_progress("QA_SERVING_RUNTIME_APPLY", serving_runtime, QA_NAMESPACE)
        if not _apply_yaml_document(sr_body, log, timeout=180):
            return _return_last_qa_error(log)

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
        _qa_progress("QA_MODEL_START", model_name, isvc_name)
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
        resource_pick = True
        fixed_res: tuple[str, str, str, str] | None = None
        last_llm_summary = ""
        image_pull_attempts = 0

        for attempt in range(max_heal_retries + 1):
            if resource_pick:
                cpu_req, mem_req, cpu_lim, mem_lim = pick_cpu_memory(
                    required_vram_gb=vram,
                    heal_bump=mem_bump,
                )
            else:
                cpu_req, mem_req, cpu_lim, mem_lim = fixed_res or pick_cpu_memory(
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
                _qa_progress("QA_MODEL_FAIL", model_name, f"reason={failure_reason}")
                break
            _qa_progress("QA_MODEL_APPLIED", model_name, isvc_name)

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
                url = resolve_inference_base_url(isvc_name, QA_NAMESPACE, log)
                _qa_progress("QA_MODEL_READY", model_name, isvc_name, url)
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
            log_si = _fetch_pod_logs(
                isvc_name,
                "storage-initializer",
                log,
                tail_lines=700,
            )
            log_ks = _fetch_pod_logs(
                isvc_name,
                "kserve-container",
                log,
                tail_lines=700,
            )
            if log_si:
                log_snip += log_si + "\n"
            if log_ks:
                log_snip += log_ks + "\n"
            pod_json_for_oom = (
                rp.stdout if rp.returncode == 0 and (rp.stdout or "").strip() else None
            )
            if logs_hint_oom(log_snip, pods_json_stdout=pod_json_for_oom):
                kind = "oom"

            if kind == "image_pull":
                if image_pull_attempts < _image_pull_retries():
                    image_pull_attempts += 1
                    logger.warning(
                        "Image pull failure for %s (attempt %d/%d), retrying in 30s",
                        model_name,
                        image_pull_attempts,
                        _image_pull_retries(),
                    )
                    time.sleep(30)
                    continue
                outcomes.append(f"{model_name}:QA_ERROR:IMAGE_PULL")
                _qa_progress(
                    "QA_MODEL_FAIL",
                    model_name,
                    "reason=image_pull",
                    f"detail={detail[:400]}",
                )
                break

            if attempt >= max_heal_retries:
                fr = detail or kind or "unknown"
                if last_llm_summary:
                    fr = f"{fr}; summary={last_llm_summary[:500]}"
                failure_reason = fr
                outcomes.append(f"{model_name}:FAIL:{failure_reason}")
                _qa_progress(
                    "QA_MODEL_FAIL",
                    model_name,
                    f"reason={detail or kind or 'unknown'}",
                    f"summary={last_llm_summary[:400]}" if last_llm_summary else f"detail={detail[:400]}",
                )
                break

            heal_label = "heuristic"
            plan_summary = ""

            if llm is not None:
                pod_excerpt = _pod_json_excerpt(isvc_name, log)
                events_tail = _fetch_recent_events(log)
                max_g = _max_gpu_allowed()
                ctx = {
                    "model_name": model_name,
                    "isvc_name": isvc_name,
                    "wait_detail": detail,
                    "pod_json_excerpt": pod_excerpt,
                    "events_tail": events_tail,
                    "logs_storage_initializer": log_si,
                    "logs_kserve_container": log_ks,
                    "current_args_json": json.dumps(cur_args),
                    "current_resources_json": json.dumps(
                        {
                            "cpu_request": cpu_req,
                            "memory_request": mem_req,
                            "cpu_limit": cpu_lim,
                            "memory_limit": mem_lim,
                            "gpu_count": gpu_n,
                        }
                    ),
                    "gpu_provider": gpu_provider,
                    "max_gpu_allowed": max_g,
                }
                plan = propose_remediation(
                    llm,
                    context=ctx,
                    fallback_args=list(cur_args),
                    fallback_cpu_req=cpu_req,
                    fallback_mem_req=mem_req,
                    fallback_cpu_lim=cpu_lim,
                    fallback_mem_lim=mem_lim,
                    fallback_gpu=gpu_n,
                )
                if plan is not None:
                    heal_label = "llm"
                    cur_args = list(plan.serving_arguments)
                    fixed_res = (
                        plan.cpu_request,
                        plan.memory_request,
                        plan.cpu_limit,
                        plan.memory_limit,
                    )
                    resource_pick = False
                    plan_gpu = plan.gpu_count
                    if gpu_provider.upper() in ("CPU", "NONE", ""):
                        plan_gpu = 0
                    gpu_n = plan_gpu
                    plan_summary = plan.summary
                    last_llm_summary = plan.summary

            if heal_label == "heuristic":
                mem_bump += 1
                resource_pick = True
                fixed_res = None
                cur_args = halve_max_model_len_args(cur_args)

            _append_report(
                log,
                f"Heal ({heal_label}) after attempt {attempt + 1}/{max_heal_retries + 1}: {detail}",
            )
            heal_fields = [
                f"attempt={attempt + 1}",
                f"mode={heal_label}",
                f"reason={detail[:300]}",
            ]
            if plan_summary:
                heal_fields.append(f"summary={plan_summary[:400]}")
            _qa_progress("QA_MODEL_HEAL", model_name, *heal_fields)
            continue

        if ok_model:
            if not _env_truthy("QA_SKIP_POST_DEPLOY_SMOKE"):
                base_url = resolve_inference_base_url(isvc_name, QA_NAMESPACE, log)
                model_id = os.environ.get("QA_SMOKE_MODEL_ID", "").strip() or isvc_name
                user_msg = os.environ.get(
                    "QA_SMOKE_USER_MESSAGE",
                    "Reply with one short sentence confirming the endpoint works.",
                ).strip()
                try:
                    max_tok = int(os.environ.get("QA_SMOKE_MAX_TOKENS", "256"))
                except ValueError:
                    max_tok = 256
                try:
                    smoke_timeout = float(os.environ.get("QA_SMOKE_TIMEOUT_S", "300"))
                except ValueError:
                    smoke_timeout = 300.0
                tls_insecure = _env_truthy("QA_SMOKE_TLS_INSECURE") or (
                    "QA_SMOKE_TLS_VERIFY" in os.environ
                    and not _env_truthy("QA_SMOKE_TLS_VERIFY")
                )
                tls_ca_file = os.environ.get("QA_SMOKE_TLS_CA_FILE", "").strip() or None
                if not base_url:
                    outcomes[-1] = f"{model_name}:SMOKE_FAIL:no_inference_url"
                    ok_model = False
                    _qa_progress(
                        "QA_MODEL_SMOKE_FAIL",
                        model_name,
                        "reason=no_inference_url",
                    )
                else:
                    _append_report(
                        log,
                        f"Smoke test POST {base_url}/v1/chat/completions model={model_id!r}",
                    )
                    smoke_ok, smoke_detail = post_chat_completions_smoke(
                        base_url,
                        model_id=model_id,
                        user_message=user_msg,
                        max_tokens=max_tok,
                        timeout_s=smoke_timeout,
                        log=log,
                        tls_ca_file=tls_ca_file,
                        tls_insecure=tls_insecure,
                    )
                    if smoke_ok:
                        _qa_progress(
                            "QA_MODEL_SMOKE_OK",
                            model_name,
                            isvc_name,
                            f"preview={smoke_detail[:200]}",
                        )
                    else:
                        outcomes[-1] = f"{model_name}:SMOKE_FAIL:{smoke_detail[:400]}"
                        ok_model = False
                        _qa_progress(
                            "QA_MODEL_SMOKE_FAIL",
                            model_name,
                            f"detail={smoke_detail[:300]}",
                        )
            if ok_model and not _env_truthy("QA_SKIP_SCALE_TO_ZERO"):
                patch_isvc_scale_to_zero(isvc_name, QA_NAMESPACE, log)
                _qa_progress("QA_MODEL_SCALED_ZERO", model_name, isvc_name)

    bad = [
        x
        for x in outcomes
        if "QA_ERROR" in x or ":FAIL:" in x or "SMOKE_FAIL" in x
    ]
    summary = "; ".join(outcomes)
    if bad:
        return "QA_ERROR:KSERVE_DEPLOYMENT_FAILED " + summary + "\n" + "\n".join(log)

    if not _env_truthy("QA_SKIP_NAMESPACE_DELETE"):
        if not delete_namespace(QA_NAMESPACE, log):
            return (
                "QA_ERROR:NAMESPACE_DELETE_FAILED "
                + summary
                + "\n"
                + "\n".join(log)
            )
        _qa_progress("QA_NAMESPACE_DELETED", QA_NAMESPACE)

    return "QA_OK:" + summary + "\n" + "\n".join(log)
