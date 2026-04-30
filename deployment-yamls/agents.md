# Deployment agent playbook (KServe QA)

Namespace: **`model-validation`**. InferenceServices are applied here after secrets.

## Required substitutions (before `oc apply`)

| Placeholder | Source |
|-------------|--------|
| Registry host | User / `REGISTRY_HOST` |
| OCI pull secret | User / `OCI_REGISTRY_PULL_SECRET` — **base64-encoded `.dockerconfigjson`** for `Secret.data` (same as ODH flow; do not double-encode) |
| vLLM / runtime image | Accelerator JSON `vllm_runtime_image` or `VLLM_RUNTIME_IMAGE` |
| ServingRuntime name | Template `__SERVING_RUNTIME_NAME__` — must exist on cluster |

## Allowed `oc` subcommands (automation)

Subprocess tooling may invoke only: `get`, `apply`, `create`, `delete`, `patch`, `logs`, `wait`, `project`, `describe`, `whoami`, `version`.

## Apply order

1. Ensure namespace exists (`model-validation`).
2. Apply OCI registry secret manifest (pull secret for model/OCI images).
3. For each model (ordered **smallest container image first** via `model_size_gb`):
   - Render `inference-service.yaml.template` with model image, args, resources.
   - `oc apply -f` the InferenceService.
   - Wait for Ready or failure.

## Monitoring

- **Events:** `oc get events -n model-validation --sort-by=.lastTimestamp`
- **ISVC:** `oc get inferenceservice -n model-validation` and conditions Ready.
- **Pods:** list pods for the ISVC label `serving.kserve.io/inferenceservice=<name>`.
- **Logs:** `oc logs <pod> -n model-validation -c storage-initializer --tail=200` and `oc logs ... -c kserve-container --tail=200` (container names may vary slightly by runtime).

## Failure signatures and remediation hints

| Signal | Likely cause | Heal direction |
|--------|----------------|----------------|
| `OOMKilled` / exit 137 | Memory too low | Raise memory limits/requests; reduce `--max-model-len` in args |
| `CrashLoopBackOff` | Bad args / runtime | Inspect logs; tune serving args; verify `storageUri` and pull secret |
| `ImagePullBackOff` | Auth / wrong image | Verify OCI secret and registry host; not auto-fixed by memory tuning |
| `Pending` + PVC | Storage | Check PVC / SC; may require cluster change |
| Ready=False long-running | Slow pull | Wait longer; check initializer logs |

## Self-heal loop (bounded)

On recoverable infra failures (OOM / memory pressure), retry with increased memory and optionally lowered `--max-model-len`. Cap retries (e.g. 3). Stop on auth/image pull errors.
