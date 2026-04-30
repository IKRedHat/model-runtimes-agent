# Runtimes Deployment Agent

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-blueviolet)](https://github.com/astral-sh/uv)
[![LangChain](https://img.shields.io/badge/Built%20with-LangChain%201.x-orange)](https://python.langchain.com)

![Supervisor Diagram](assets/supervisor-diagram.png)

Supervisor-driven orchestration for analysing model-car configurations and validating them end-to-end. The tool follows LangChain’s [supervisor pattern](https://docs.langchain.com/oss/python/langchain/supervisor), combining a primary LLM with configuration + accelerator + decision specialists and a QA runner that understands Red Hat “model-car” manifests and the container images they reference.

## Features

- **CLI-first workflow** – install the package and run `agent configuration --config …` to query any model-car file.
- **LangChain supervisor** – the `LLMAgent` composes a specialist registry (configuration + accelerator + deployment decision) and exposes a single `run_supervisor` entry point.
- **Configuration specialist** – parses YAML, surfaces serving arguments, GPU counts, parameter/quantization hints from model names, and estimates per-model VRAM.
- **Accelerator specialist** – checks OpenShift authentication, enumerates GPUs, writes a detailed `info/gpu_info.txt` report (provider, instance type, per-GPU VRAM), and returns machine-readable metadata (GPU availability, provider, and a recommended `vllm_runtime_image` pulled from OpenShift templates) so downstream steps use the exact runtime container the cluster expects.
- **Decision specialist** – reads the cached requirements plus `info/gpu_info.txt`, compares each model's VRAM requirement to the per-GPU memory, and issues a GO/NO-GO verdict.
- **Quantization-aware decisioning** – the Decision Specialist now enforces a hardware/quantization compatibility matrix (AWQ, GPTQ, FP8, W4A16/W8A8, GGUF, bitsandbytes, etc.) against the detected GPU generation, flagging unsupported kernels and recommending safer variants before issuing a verdict.
- **Serving-argument optimizer** – when the Decision Specialist emits `OPTIMIZED_SERVING_ARGUMENTS_JSON`, the Configuration Specialist applies it to `config-yaml/sample_modelcar_config.base.yaml` and writes the merged result to `config-yaml/sample_modelcar_config.generated.yaml` so QA/testing reuse the tuned flags without mutating the base template.
- **Container metadata enrichment** – shells out to `skopeo inspect` to capture aggregate image size (GB) and supported CPU architecture per model; reports image size separately from VRAM.
- **QA specialist** – applies `deployment-yamls/` templates to namespace `model-validation` (registry secret + per-model InferenceServices), deploys models in **ascending image size** order, watches readiness and key container logs, retries with higher memory and adjusted `--max-model-len` on bounded failures, and reports `QA_OK` / `QA_ERROR:<reason>` to the supervisor.
- **Checklist-style responses** – specialists start with a short checklist of the steps they are taking before returning the report/results.
- **Config bootstrap** – pass a bootstrap file at agent creation time so repeated prompts reuse cached requirements.

## Requirements

- Python **3.12+**
- A Google Gemini API key (`GEMINI_API_KEY`) for `langchain-google-genai`
- `skopeo` available on `PATH` (for container metadata; falls back gracefully if missing)
- Environment for QA: set `REGISTRY_HOST`, `OCI_REGISTRY_PULL_SECRET` (base64 `.dockerconfigjson`, or raw JSON which is encoded once), ensure `KUBECONFIG` points to a reachable cluster (defaults to `~/.kube/config`), optionally `VLLM_RUNTIME_IMAGE`, `KSERVE_SERVING_RUNTIME_NAME`, and `KSERVE_MODEL_FORMAT` to match your cluster’s ServingRuntime
- Dependencies listed in `pyproject.toml`

## Installation

```bash
# Editable install while iterating
uv pip install -e .

# or with pip
pip install -e .
```

## Streamlit UI

Run the Streamlit app locally:

```bash
streamlit run frontend/app.py
```

The app opens in your browser and prompts for the API key, pull secret, and YAML upload in the sidebar.

## CLI Usage

### Required inputs

1. **API key** – export `GEMINI_API_KEY` (required even for configuration-only runs).
2. **Model-car file** – pass `--config /path/to/modelcar.yaml` (defaults to `config-yaml/sample_modelcar_config.yaml`; use the `*.base.yaml` template or any custom file).
3. **LLM choice (optional)** – override `--model` if you want something other than `gemini-2.5-pro`.
4. **QA prerequisites (only if you expect the supervisor to run QA):**
   - `REGISTRY_HOST` – OCI registry hostname (or set per run with `--registry-host` / inferrable as a single host from the model-car).
   - `OCI_REGISTRY_PULL_SECRET` – base64 `.dockerconfigjson` (or raw JSON; encoded once for the Secret).
   - `KUBECONFIG` – path to a valid OpenShift kubeconfig (defaults to `~/.kube/config`).
   - `VLLM_RUNTIME_IMAGE` – optional override of the vLLM runtime image (otherwise the accelerator report supplies the correct image).

```bash
export GEMINI_API_KEY="your-key-here"

# Inspect the bundled base configuration (never overwritten)
agent --config config-yaml/sample_modelcar_config.base.yaml

```

The CLI still defaults to `config-yaml/sample_modelcar_config.yaml`; pass `--config` explicitly to use either the immutable base template or the generated overlay the supervisor produces when it optimises serving arguments. The generated file lives alongside the base YAML so you can diff, commit, or hand the tuned parameters back to an operator. Use `--model` if you want to run the supervisor with a different Gemini model (defaults to `gemini-2.5-pro`).

The command prints a **Configuration** section containing the parsed model requirements, including container size, estimated VRAM (based on parameters/quantization parsed from the model name), and supported architecture. Example snippet:

```json
{
  "granite-3.1-8b-instruct": {
    "model_name": "granite-3.1-8b-instruct",
    "image": "oci://registry.redhat.io/rhelai1/modelcar-granite-3-1-8b-instruct:1.5",
    "gpu_count": 1,
    "arguments": [
      "--uvicorn-log-level=info",
      "--max-model-len=2048",
      "--trust-remote-code",
      "--distributed-executor-backend=mp"
    ],
    "model_size_gb": 15.23,
    "model_p_billion": 8.0,
    "quantization_bits": 16,
    "required_vram_gb": 18,
    "supported_arch": "amd64"
  }
}
```

### Sample run (with checklists and decision step)

```
agent --config config-yaml/sample_modelcar_config.base.yaml
```

Produces four sections:

- Configuration
  - [x] Load cached requirements
  - [x] Estimate VRAM
  - [x] Compose report
  - Model: granite-3.1-8b-instruct → 15.24 GB image, 8B params, not quantized, ~18 GB VRAM, arch amd64
- Accelerator Compatibility
  - [x] Load cached requirements
  - [x] Check cluster authentication
  - [x] Query GPU status
  - [x] Fetch detailed GPU info (written to `info/gpu_info.txt`)
  - [x] Validate accelerator compatibility (reports provider, status, and CUDA/ROCm notes)
  - Includes a validation table summarising authentication, GPU availability, and whether the detected GPU meets the model’s VRAM requirement.
  - Surfaces machine-readable metadata (`gpu_available`, `gpu_provider`, `vllm_runtime_image`) that the supervisor later feeds into QA so validation uses the same runtime image OpenShift templates recommend.
- Deployment Decision
  - [x] Load cached requirements
  - [x] Read GPU info file
  - [x] Compare VRAM needs vs per-GPU memory (e.g., `model VRAM 18 GB vs GPU 80 GB`)
  - [x] Issue GO/NO-GO recommendation with justification
- QA Validation
  - [x] Stage kubeconfig + OCI pull secret into a temporary directory
  - [x] Copy `config-yaml/sample_modelcar_config.generated.yaml` into the QA container
  - [x] Launch `quay.io/opendatahub/opendatahub-tests:latest` with Podman and stream `[QA] …` logs
  - [x] Summarize the run with `QA_OK` or `QA_ERROR:<reason>` plus remediation guidance

Example supervisor transcript (captured in `demo-logs/success-log-example4.log`):

```markdown
### Configuration Summary
The Configuration Specialist reported on four models preloaded in the model-car, with a total estimated VRAM requirement of 22 GB. The primary model for this assessment, `Llama-3.1-8B-Instruct`, requires an estimated 18 GB of VRAM.

### Accelerator Summary
The Accelerator Specialist confirmed that the cluster is healthy and accessible. Authentication was successful, and the cluster has NVIDIA GPUs available that are compatible with CUDA and vLLM, meeting the deployment requirements.

### Deployment Decision
The Decision Specialist has issued a **GO** for this deployment.

The specialist's reasoning is as follows:
- **GPU Capacity**: The available GPU VRAM (44.99 GB) is sufficient to meet the model's estimated requirement of 18 GB.
- **Serving-argument Suitability**: The initial serving arguments were found to be suboptimal for a single-GPU environment. The Decision Specialist recommended removing an unnecessary distributed executor flag and explicitly setting `tensor_parallel_size=1`.
- **Environment / Access Health**: The Accelerator Specialist reported no authentication or connectivity failures.

Based on the specialist's recommendation, the optimized serving arguments were written back to the model-car configuration by the Configuration Specialist. The applied optimization was:
```json
{
  "args": [
    "--uvicorn-log-level=debug",
    "--max-model-len=1024",
    "--trust-remote-code",
    "--tensor-parallel-size=1"
  ]
}
```

### QA Validation
The QA Specialist ran the Opendatahub model validation test suite against the updated configuration. The result was a **PASS**, with all 4 validation tests completing successfully. This confirms that the model serving deployments are correctly configured and operational.
```

Behind the scenes the Decision Specialist now cross-checks quantization hints (e.g., `w4a16`, `fp8`, AWQ/GPTQ/GGUF naming) against the detected accelerator generation (Volta/Turing/Ampere/Ada/Hopper, AMD, Intel, etc.). Any mismatch is surfaced as a compatibility warning or a forced **NO-GO**, preventing you from deploying a kernel that the GPU simply cannot execute.

A typical end-to-end response therefore concludes with something like:

> **Deployment Decision**
> - `granite-3.1-8b-instruct` needs 18 GB VRAM
> - GPU inventory reports NVIDIA H100 (80 GB each), 8 GPUs total
> - Comparison: `model VRAM 18 GB vs GPU 80 GB`
> - **GO**: one GPU is enough, leaving ample capacity in the cluster

- When QA runs, look for `[QA] …` streaming logs and a closing `QA_OK:` or `QA_ERROR:<reason>` line in the **QA Validation** section to confirm KServe deployment validation results.
- Use `--config` to point at any other YAML file (base template, generated overlay, or your own manifest).
- `LLMAgent` also accepts a `bootstrap_config` parameter if you embed it in your own Python application.

### Applying optimized serving arguments

1. The Decision Specialist inspects the `serving_arguments` it inherited from the configuration summary. If any flag is missing/unsafe (e.g., no `--tensor-parallel-size`), it returns an `OPTIMIZED_SERVING_ARGUMENTS_JSON` block that contains the minimal fix(es).
2. The supervisor relays that JSON to the Configuration Specialist, which calls `generate_optimal_serving_arguments(optimized_args_json=<blob>)`.
3. That tool uses `config-yaml/sample_modelcar_config.base.yaml` when present (falling back to `config-yaml/sample_modelcar_config.yaml`), merges the overrides, and writes the result to `config-yaml/sample_modelcar_config.generated.yaml`.
4. The tool’s response (`Updated serving arguments for model(s): … Generated config: …`) is echoed directly in the supervisor transcript so you can confirm what was touched.

Because the base template is never mutated you can always diff the generated overlay, commit it, or pass it straight to the QA pipeline.

## QA Validation

The QA Specialist applies manifests under [`deployment-yamls/`](deployment-yamls/) (registry Secret + InferenceService template), writes substituted values for `REGISTRY_HOST`, `OCI_REGISTRY_PULL_SECRET`, and the vLLM runtime image, deploys **deployable** models from `info/deployment_matrix.json` that appear in `config-yaml/sample_modelcar_config.generated.yaml`, ordered by container image size (`model_size_gb`). It only runs after the accelerator step reports a healthy cluster.

- Uses allowlisted `oc` commands (`get`, `apply`, `create`, `delete`, `patch`, `logs`, `wait`, …) against namespace **`model-validation`**.
- Waits for InferenceService Ready, inspects pod status for OOM / image pull failures, optionally tails `storage-initializer` and `kserve-container` logs, and retries with increased memory and halved `--max-model-len` up to a bounded number of attempts (except on image-pull/auth failures).
- Final output begins with `QA_OK:` or `QA_ERROR:` (`QA_ERROR:REGISTRY_HOST_MISSING`, `QA_ERROR:KSERVE_DEPLOYMENT_FAILED`, etc.).

When the accelerator metadata is healthy it emits JSON containing `vllm_runtime_image`. The supervisor forwards that image to the QA Specialist. You may override with `VLLM_RUNTIME_IMAGE` or `--vllm-runtime-image`.

Environment variables:

```bash
export REGISTRY_HOST="registry.redhat.io"       # REQUIRED unless inferred as a single host from model-car
export OCI_REGISTRY_PULL_SECRET="..."           # REQUIRED: base64 .dockerconfigjson (or raw JSON)
export KUBECONFIG="$HOME/.kube/config"          # Optional (defaults to ~/.kube/config)
export VLLM_RUNTIME_IMAGE="quay.io/modh/..."    # Optional; accelerator value used if unset
export KSERVE_SERVING_RUNTIME_NAME="vllm-runtime" # Optional — must exist on cluster
export KSERVE_MODEL_FORMAT="huggingface"          # Optional predictor.model.modelFormat.name
export QA_PER_MODEL_TIMEOUT_S="900"               # Optional wait budget per model (seconds)
```

## Configuration Files

The repo ships two sample manifests:

- `config-yaml/sample_modelcar_config.base.yaml` – canonical template referenced throughout the README; update this when you want to change defaults.
- `config-yaml/sample_modelcar_config.generated.yaml` – auto-generated overlay containing the latest optimized `serving_arguments` from the supervisor loop (safe to edit/commit).

Each file still contains the same schema:

- `model-car`: list (or single mapping) describing each model. Keys:
  - `name`: identifier reported in CLI output.
  - `image`: OCI reference (transport prefixes such as `oci://` are supported).
  - `serving_arguments.gpu_count`: advertised GPU count; surfaced alongside metadata.
  - `serving_arguments.args`: extra runtime flags (e.g., `--max-model-len`).
- `default`: optional fallback block; currently used only as documentation.

## Architecture

```
src/runtimes_dep_agent/
├── agent/
│   ├── llm_agent.py              # Supervisor wiring + specialist registry
│   └── specialists/
│       ├── config_specialist.py
│       ├── accelerator_specialist.py
│       ├── decision_specialist.py
│       └── qa_specialist.py      # KServe deployment QA + qa_kserve pipeline
├── config/
│   └── model_config.py           # YAML + skopeo helpers
├── execute_agent.py              # CLI entry point
├── reports/
│   └── validation_report.py      # Future reporting hooks
└── validators/
    └── accelerator_validator.py
```

- Packages live under `src/runtimes_dep_agent`; installing the project exposes the console script `agent`.
- The specialist tooling is deliberately modular—drop another specialist builder into `agent/specialists/` and register it inside `LLMAgent._initialise_specialists`.

## Development Notes

- Run `python -m compileall src` for a quick syntax check; add pytest suites under `tests/` as functionality grows.
- Regenerate the CLI entry point after edits with `pip install -e .` (the `agent` command resolves to `runtimes_dep_agent.execute_agent:main`).
- When `skopeo` cannot inspect an image (permissions, offline, etc.), the tool falls back to `0` size and `unknown` architecture while still printing other requirements.

## License

Licensed under the [Apache License 2.0](LICENSE).
