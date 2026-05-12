# Deployment agent playbook (KServe QA)

Namespace: **`model-validation`**. InferenceServices are applied here after secrets.

## Required substitutions (before `oc apply`)

| Placeholder | Source |
|-------------|--------|
| Registry host | User / `REGISTRY_HOST` |
| OCI pull secret | User / `OCI_REGISTRY_PULL_SECRET` — **base64-encoded `.dockerconfigjson`** for `Secret.data` (same as ODH flow; do not double-encode) |
| vLLM / runtime image | QA tool / `VLLM_RUNTIME_IMAGE` if set; otherwise image from cluster `oc get template` in `redhat-ods-applications` (CUDA template for `NONE`/`CPU`/unknown; provider-specific for NVIDIA/AMD/Spyre/Intel) |
| ServingRuntime | QA applies `serving-runtime.yaml.template` before InferenceServices (`metadata.name` = `KSERVE_SERVING_RUNTIME_NAME`, default `vllm-runtime`). Set `QA_SKIP_SERVING_RUNTIME_APPLY=1` to skip if the runtime already exists. |

## Allowed `oc` subcommands (automation)

Subprocess tooling may invoke only: `get`, `apply`, `create`, `delete`, `patch`, `logs`, `wait`, `project`, `describe`, `whoami`, `version`.

## Apply order

1. Ensure namespace exists (`model-validation`).
2. Apply OCI registry secret manifest (pull secret for model/OCI images).
3. Render and apply `serving-runtime.yaml.template` (vLLM image from `VLLM_RUNTIME_IMAGE`, name from `KSERVE_SERVING_RUNTIME_NAME`, model format from `KSERVE_MODEL_FORMAT` for `supportedModelFormats`) unless `QA_SKIP_SERVING_RUNTIME_APPLY` is set.
4. For each model (ordered **smallest container image first** via `model_size_gb`):
   - Render `inference-service.yaml.template` with model image, args, resources.
   - `oc apply -f` the InferenceService.
   - Wait for Ready or failure.
   - Resolve external/base URL (`status.url` / predictor URL / OpenShift Route), then **POST** `/v1/chat/completions` (OpenAI-style) unless `QA_SKIP_POST_DEPLOY_SMOKE=1`. TLS verification is **off** by default (like `curl -k`); set `QA_SMOKE_TLS_VERIFY=1` to enforce certs.
   - **Scale to zero** (`minReplicas`/`maxReplicas` patch) unless `QA_SKIP_SCALE_TO_ZERO=1`.
5. If every model succeeds (including smoke when enabled), **delete namespace** `model-validation` unless `QA_SKIP_NAMESPACE_DELETE=1`.

**Smoke test env (optional):** `QA_SMOKE_MODEL_ID` (default: InferenceService name), `QA_SMOKE_USER_MESSAGE`, `QA_SMOKE_MAX_TOKENS`, `QA_SMOKE_TIMEOUT_S`.

## Monitoring

- **Events:** `oc get events -n model-validation --sort-by=.lastTimestamp`
- **ISVC:** `oc get inferenceservice -n model-validation` and conditions Ready.
- **Pods:** list pods for the ISVC label `serving.kserve.io/inferenceservice=<name>`.
- **Logs:** automation tails **storage-initializer** and **kserve-container** with a larger tail (hundreds of lines). Manual checks often use `--tail=200`.

## Failure signatures and remediation hints

| Signal | Likely cause | Heal direction |
|--------|----------------|----------------|
| `OOMKilled` / exit 137 | Memory too low | Raise memory limits/requests; reduce `--max-model-len` in args |
| `CrashLoopBackOff` | Bad args / runtime | Inspect logs; tune serving args; verify `storageUri` and pull secret |
| `ImagePullBackOff` | Auth / wrong image | Verify OCI secret and registry host; not auto-fixed by memory tuning |
| `Pending` + PVC | Storage | Check PVC / SC; may require cluster change |
| Ready=False long-running | Slow pull | Wait longer; check initializer logs |

## Self-heal loop (bounded, LLM-driven when available)

- **Attempts:** at most **3 total deploys per model** (1 initial + **2** remediation redeploys). Tune via `max_heal_retries=2` in code (i.e. `max_heal_retries + 1` attempts).
- **v1 patch surface:** **serving arguments** (full replacement list) and **CPU / memory requests & limits** and **GPU count** only (no PVC/volume edits until templates expose optional volume placeholders).
- **Image pull / registry auth:** classified as **non-recoverable** by automation — **no LLM retry**; emit failure with `reason=image_pull`.
- **LLM output (JSON only):** the remediation model must return a single object shaped like:

```json
{
  "summary": "short root cause for operators",
  "serving_arguments": ["--tensor-parallel-size=2", "--max-model-len=4096"],
  "resources": {
    "cpu_request": "6",
    "memory_request": "16Gi",
    "cpu_limit": "10",
    "memory_limit": "20Gi",
    "gpu_count": 2
  }
}
```

- **GPU cap:** `gpu_count` is clamped to **`QA_MAX_GPU_COUNT`** (default **8**). CPU-only providers force **0** GPUs regardless of the proposal.
- **Heuristic fallback:** if no LLM is configured or JSON parse/validation fails, heal uses memory-tier bump + halved `--max-model-len` (previous behavior).
- **Telemetry:** progress lines may include `QA_MODEL_HEAL::...::summary=...` and final `QA_MODEL_FAIL::...` includes the last readiness signal and any remediation summary.
