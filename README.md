# Model Runtimes Deployment Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

A **supervisor-style LangChain agent** that evaluates Red Hat **model-car** YAML against your OpenShift cluster, produces deterministic **deploy / no-deploy** guidance, optionally **optimizes serving arguments**, and—when the cluster is healthy—runs **end-to-end KServe validation** using `oc` and rendered manifests (not legacy in-cluster test containers).

![Supervisor diagram](assets/supervisor-diagram.png)

---

## What this project does today

| Previously | Now |
|------------|-----|
| Separate validation flows and ad hoc cluster checks | One **supervisor** coordinates four **specialists** behind a single workflow |
| Heavy reliance on external “validation suite” containers for QA | **KServe QA** in namespace `model-validation`: registry Secret, ServingRuntime, InferenceServices, readiness, optional smoke `POST /v1/chat/completions`, bounded self-heal, scale-to-zero |
| Informal deployment opinions | **`info/deployment_matrix.json`** from a **deterministic deployability engine** (quantization vs GPU generation, VRAM, etc.) so GO/NO-GO aligns with machine-readable facts |

Operational detail for KServe apply order, allowed `oc` verbs, and environment toggles lives in **[`deployment-yamls/agents.md`](deployment-yamls/agents.md)**.

---

## Specialist agents (what each one is responsible for)

All specialists share the same Google **Gemini** chat model instance (temperature 0). The **supervisor** (`LLMAgent`) decides which tools to call and assembles the final structured report (Configuration → Accelerator → Deployment Decision → QA).

### 1. Configuration Specialist

- **Role:** Interpret the **preloaded** model-car (parsed at startup from your YAML path). Answer questions about per-model **images**, **serving arguments**, **GPU count**, **estimated VRAM**, **container size** (`skopeo` when available), and **architecture**.
- **Key behaviour:** When the Decision specialist proposes **`OPTIMIZED_SERVING_ARGUMENTS_JSON`**, this specialist can **merge** those changes and write **`config-yaml/sample_modelcar_config.generated.yaml`**, keeping only models marked **deployable** in **`info/deployment_matrix.json`** (and creating `config-yaml/` if needed).
- **Typical tools:** Summaries of preloaded requirements, VRAM inference, **`generate_optimal_serving_arguments`**.

### 2. Accelerator Specialist

- **Role:** Treat the cluster as the source of truth for **authentication**, **GPU / accelerator inventory**, and the **vLLM runtime image** the cluster templates expect.
- **Key outputs:** Writes **`info/gpu_info.txt`**. Exposes JSON metadata including **`vllm_runtime_image`** (or **`vllm_image`** in nested metadata, depending on path) for the supervisor to pass unchanged into QA.
- **Safety:** If login or connectivity fails, the supervisor is instructed to declare **NO-GO** and **not** run QA.

### 3. Decision Specialist

- **Role:** **GO / NO-GO** narrative that must stay consistent with hardware and policy.
- **Authoritative artefact:** Persists **`info/deployment_matrix.json`** using **`deployability_engine`** (per-model `deployable` and `reason`), informed by **`info/gpu_info.txt`** and cached model requirements. Also writes **`info/deployment_info.txt`** for prose reporting.
- **Extras:** Tensor-parallel sizing hints from GPU memory vs weight proxies; may emit structured **optimized serving arguments** for the Configuration specialist.

### 4. QA Specialist

- **Role:** Run **`run_kserve_deployment_qa`**—render templates under **`deployment-yamls/`**, apply manifests in **`model-validation`**, wait for InferenceService readiness, tail relevant logs, apply **bounded remediation** (memory, `--max-model-len`, optional LLM-driven patches where configured), then optional **smoke inference**, **scale to zero**, and **namespace cleanup** on full success.
- **Inputs:** Deployable models from **`deployment_matrix.json`** that appear in **`sample_modelcar_config.generated.yaml`**, ordered by **smallest container image first** (`model_size_gb`). Registry host from **`REGISTRY_HOST`** or, when unambiguous, inferred from a single registry in the model-car images.
- **Output:** A string beginning with **`QA_OK:`** or **`QA_ERROR:`** for automation and the supervisor transcript.

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| **Python 3.12+** | Declared in `pyproject.toml`. |
| **[uv](https://github.com/astral-sh/uv)** (recommended) | Installs dependencies and runs `streamlit` / `agent` without activating a venv manually. |
| **`oc`** | OpenShift / Kubernetes CLI; required for pre-flight and QA. |
| **`skopeo`** | Required for pre-flight; used to enrich container metadata when images are inspectable. |
| **`GEMINI_API_KEY`** | Google AI key for `langchain-google-genai` (supervisor + specialists). |
| **Valid kube context** | `KUBECONFIG` or default `~/.kube/config` for accelerator and QA steps. |
| **Registry credentials for QA** | `REGISTRY_HOST` and `OCI_REGISTRY_PULL_SECRET` when you expect the QA specialist to apply pulls (see [Environment variables](#environment-variables)). |

Python dependencies include LangChain 1.x, Streamlit, PyYAML, Pandas, Plotly, and Pillow—see **`pyproject.toml`** for exact ranges.

---

## Installation

Clone the repository and install in editable mode (pick one).

**Using uv (recommended):**

```bash
cd model-runtimes-agent
uv sync
# or, if you prefer pip inside uv:
uv pip install -e .
```

**Using pip:**

```bash
pip install -e .
```

This exposes the **`agent`** console script (`runtimes_dep_agent.execute_agent:main`).

---

## How to run

### Option A — Web UI (Streamlit)

From the repository root, with dependencies installed:

```bash
uv run streamlit run app.py
```

If you use a classic virtual environment instead of `uv run`:

```bash
streamlit run app.py
```

The UI collects **Gemini API key**, **OCI pull secret**, **registry host**, and model-car input, runs the same supervisor pipeline, and writes artefacts under a per-session **`info/`** directory (`AGENT_RUN_INFO_DIR` is set automatically for subprocesses).

---

### Option B — Command-line interface (`agent`)

```bash
export GEMINI_API_KEY="your-key"

# Example: model-car at repo root (adjust path as needed)
uv run agent --config sample_modelcar_config.yaml
```

Default config path in code is **`config-yaml/sample_modelcar_config.yaml`**. Ensure that file exists or always pass **`--config`**.

**Useful CLI flags:**

| Flag | Description |
|------|-------------|
| `--config PATH` | Model-car YAML to preload (default: `config-yaml/sample_modelcar_config.yaml`). |
| `--model NAME` | Gemini model id (default: `gemini-2.5-pro`). |
| `--gemini-api-key KEY` | Sets `GEMINI_API_KEY` for this process. |
| `--oci-pull-secret SECRET` | Sets `OCI_REGISTRY_PULL_SECRET`. |
| `--registry-host HOST` | Sets `REGISTRY_HOST`. |
| `--vllm-runtime-image IMAGE` | Sets `VLLM_RUNTIME_IMAGE`. |
| `--oc-login 'oc login ...'` | Runs **`oc login` only** before the agent (full string; subcommand must be `login`). Alternative: `OC_LOGIN_COMMAND` env var. |
| `--report-output PATH` | HTML report output (default: `report.html`). |

**Without uv** (after `pip install -e .`):

```bash
agent --config sample_modelcar_config.yaml
```

The CLI runs **pre-flight** (`oc` + `skopeo`), optional **`oc login`**, streams the supervisor, syncs deployment prose with the matrix, prints the final summary, and generates **`report.html`** (or `--report-output`).

---

## Environment variables

### Always (supervisor + specialists)

| Variable | Required | Purpose |
|----------|----------|---------|
| `GEMINI_API_KEY` | **Yes** (unless passed via `--gemini-api-key` or Streamlit) | Google Generative AI authentication. |

### OpenShift login (optional helper)

| Variable | Purpose |
|----------|---------|
| `OC_LOGIN_COMMAND` | Same as `--oc-login`: full `oc login ...` string executed before the run. |

### Accelerator + QA (cluster and registry)

| Variable | When needed | Purpose |
|----------|----------------|----------|
| `KUBECONFIG` | QA / accelerator | Path to kubeconfig (defaults to `~/.kube/config`). |
| `REGISTRY_HOST` | QA (unless inferred) | OCI registry hostname for pull Secret and manifests. |
| `OCI_REGISTRY_PULL_SECRET` | QA | Base64 **`.dockerconfigjson`** body for `Secret.data`, or raw JSON (normalized by the tool). |
| `VLLM_RUNTIME_IMAGE` | Optional | Overrides image discovered from OpenShift templates; supervisor may still prefer accelerator metadata when set. |

### QA tuning (optional)

| Variable | Default / notes |
|----------|-----------------|
| `OCI_REGISTRY_SECRET_NAME` | Kubernetes Secret name for pulls (see `qa_kserve` / `agents.md`). |
| `KSERVE_SERVING_RUNTIME_NAME` | ServingRuntime resource name (e.g. `vllm-runtime`). |
| `KSERVE_MODEL_FORMAT` | Predictor model format (e.g. `huggingface`). |
| `QA_PER_MODEL_TIMEOUT_S` | Per-model wait budget (seconds). |
| `QA_MAX_GPU_COUNT` | Upper bound for remediation GPU suggestions. |
| `QA_SKIP_SERVING_RUNTIME_APPLY` | Set to `1` / `true` if ServingRuntime already exists. |
| `QA_SKIP_POST_DEPLOY_SMOKE` | Skip HTTP smoke after Ready. |
| `QA_SKIP_SCALE_TO_ZERO` | Skip scale-to-zero patch. |
| `QA_SKIP_NAMESPACE_DELETE` | Keep `model-validation` after success. |
| `QA_SMOKE_TLS_VERIFY` | Enforce TLS on smoke requests (default behaviour is permissive). |
| `QA_SMOKE_MODEL_ID`, `QA_SMOKE_USER_MESSAGE`, `QA_SMOKE_MAX_TOKENS`, `QA_SMOKE_TIMEOUT_S` | Smoke request tuning. |

### Streamlit / advanced

| Variable | Purpose |
|----------|---------|
| `AGENT_RUN_INFO_DIR` | Set by the app to pin **`info/`** for a run; CLI uses a temp directory when unset. |

---

## Configuration layout

- **Model-car YAML** — Top-level `model-car:` list with `name`, `image` (`oci://` supported), optional `serving_arguments.args` and `gpu_count`.
- **`config-yaml/sample_modelcar_config.base.yaml`** — Optional canonical base for merges when using the optimizer path.
- **`config-yaml/sample_modelcar_config.generated.yaml`** — Produced by the Configuration specialist for QA (deployable models + tuned args).

---

## Outputs

| Path | Description |
|------|-------------|
| `info/models_info.json` | Cached requirements from bootstrap YAML. |
| `info/gpu_info.txt` | Accelerator inventory text. |
| `info/deployment_matrix.json` | Deterministic deployability rows. |
| `info/deployment_info.txt` | Decision narrative. |
| `info/supervisor_summary.txt` | Final supervisor text (CLI temp run). |
| `report.html` | Self-contained HTML report (from `info/` + pre-flight). |

---

## Repository layout (source)

```
src/runtimes_dep_agent/
├── agent/llm_agent.py          # Supervisor
├── agent/specialists/          # Config, accelerator, decision, QA builders
├── config/model_config.py      # YAML + skopeo helpers
├── qa_kserve/                  # KServe pipeline, render, oc, post-deploy, remediation
├── validators/                 # Accelerator checks, deployability engine, matrix prose sync
├── report/html_report.py
├── preflight.py
├── execute_agent.py            # CLI entrypoint
app.py                          # Streamlit UI
deployment-yamls/               # Templates + operators' agents.md
```

---

## Development

```bash
uv run pytest
uv run python -m compileall src
```

---

## License

[Apache License 2.0](LICENSE).
