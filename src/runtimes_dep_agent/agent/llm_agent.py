"""Supervisor orchestration that wires specialist agents together."""

from __future__ import annotations

import re
from typing import Any, Dict, List
import logging
import json
import os
from pathlib import Path

from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langchain_google_genai import ChatGoogleGenerativeAI

from .specialists import SpecialistSpec
from .specialists.config_specialist import build_config_specialist
from .specialists.qa_specialist import build_qa_specialist
from .specialists.accelerator_specialist import build_accelerator_specialist
from .specialists.decision_specialist import build_decision_specialist
from ..config.model_config import load_llm_model_config, get_model_requirements
from ..utils.path_utils import detect_repo_root



logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Auth-failure detection
# ------------------------------------------------------------------ #

_AUTH_FAILURE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(kw, re.IGNORECASE)
    for kw in (
        r"unauthorized",
        r"forbidden",
        r"cluster login failed",
        r"unable to reach cluster",
        r"tls handshake",
        r"cannot connect",
        r"connection refused",
        r"authentication failed",
    )
]


def _accelerator_indicates_auth_failure(text: str) -> bool:
    """Return True when *text* contains keywords signalling cluster auth/connectivity failure."""
    return any(pat.search(text) for pat in _AUTH_FAILURE_PATTERNS)






class LLMAgent:
    """Builds a collection of specialists and exposes a supervisor entry point."""

    def __init__(self, 
                 api_key: str, 
                 model: str = "gemini-2.5-pro",
                 bootstrap_config: str | None = None,
                 info_dir: Path | None = None,
                 ) -> None:
        self.info_dir = info_dir
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=api_key,
            temperature=0,
        )

        self._cluster_auth_ok: bool = True
        self.precomputed_requirements = None
        self.bootstrap_config_path: Path | None = None
        if bootstrap_config:
            self.bootstrap_config_path = Path(bootstrap_config).resolve()
            config = load_llm_model_config(str(self.bootstrap_config_path))
            self.precomputed_requirements = get_model_requirements(config)
            # Save precomputed requirements to info/models_info.json
            self._save_precomputed_requirements()

        self.specialists: List[SpecialistSpec] = self._initialise_specialists()
        self._supervisor = self._create_supervisor()

    # ------------------------------------------------------------------ #
    # Supervisor operations
    # ------------------------------------------------------------------ #
    def run_supervisor(
        self, user_input: str, recursion_limit: int = 100
    ) -> Dict[str, Any]:
        """Invoke the top-level supervisor on a natural language request."""
        return self._supervisor.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": recursion_limit},
        )

    def stream_supervisor(
        self, user_input: str, recursion_limit: int = 100
    ):
        """Yield streaming events from the supervisor for live output."""
        return self._supervisor.stream(
            {"messages": [{"role": "user", "content": user_input}]},
            config={"recursion_limit": recursion_limit},
            stream_mode="updates",
        )

    def extract_final_text(self, result: Dict[str, Any]) -> str:
        """Extract the supervisor's final textual response."""
        return self._extract_final_text(result)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _initialise_specialists(self) -> List[SpecialistSpec]:
        specialists: List[SpecialistSpec] = []
        specialists.append(build_config_specialist(
            self.llm,
            self._extract_final_text,
            self.precomputed_requirements,
            bootstrap_config_path=self.bootstrap_config_path,
            info_dir=self.info_dir,
        ))
        specialists.append(build_accelerator_specialist(
            self.llm,
            self._extract_final_text,
            self.precomputed_requirements,
            info_dir=self.info_dir,
        ))
        specialists.append(build_decision_specialist(
            self.llm,
            self._extract_final_text,
            self.precomputed_requirements,
            info_dir=self.info_dir,
        ))
        specialists.append(build_qa_specialist(
            self.llm,
            self._extract_final_text,
            self.precomputed_requirements,
            info_dir=self.info_dir,
        ))
        return specialists

    def _create_supervisor(self):
        tools = [spec.tool for spec in self.specialists]

        # -------------------------------------------------------------- #
        # Wrap accelerator & QA tools for programmatic auth-failure guard #
        # -------------------------------------------------------------- #
        agent_ref = self  # capture for closures

        for idx, t in enumerate(tools):
            if getattr(t, "name", None) == "analyze_accelerator":
                original_accel = t

                @lc_tool
                def analyze_accelerator(request: str) -> str:  # noqa: E303
                    """Delegate accelerator and GPU validation requests to the accelerator specialist.
                    Always return the JSON output from get_accelerator_metadata_json() to the Supervisor.
                    """
                    result = original_accel.invoke({"request": request})
                    if _accelerator_indicates_auth_failure(str(result)):
                        agent_ref._cluster_auth_ok = False
                        logger.warning(
                            "Accelerator output indicates auth/connectivity failure; "
                            "QA will be blocked."
                        )
                    return result

                analyze_accelerator.name = "analyze_accelerator"  # type: ignore[attr-defined]
                tools[idx] = analyze_accelerator
                logger.debug("Wrapped analyze_accelerator tool with auth-failure guard.")

            elif getattr(t, "name", None) == "analyze_qa_results":
                original_qa = t

                @lc_tool
                def analyze_qa_results(  # noqa: E303
                    request: str, runtime_image: str, gpu_provider: str,
                ) -> str:
                    """Supervisor-facing entrypoint. Pass:
                        - request: e.g. "Run QA and summarize validation results."
                        - runtime_image: vLLM runtime image (from accelerator JSON / VLLM_RUNTIME_IMAGE).
                        - gpu_provider: e.g. NVIDIA or AMD.
                    """
                    if not agent_ref._cluster_auth_ok:
                        msg = (
                            "QA_ERROR:CLUSTER_AUTH_FAILED Accelerator reported "
                            "authentication/connectivity failure; QA skipped."
                        )
                        logger.warning(msg)
                        return msg
                    return original_qa.invoke(
                        {
                            "request": request,
                            "runtime_image": runtime_image,
                            "gpu_provider": gpu_provider,
                        }
                    )

                analyze_qa_results.name = "analyze_qa_results"  # type: ignore[attr-defined]
                tools[idx] = analyze_qa_results
                logger.debug("Wrapped analyze_qa_results tool with auth-failure guard.")

        runtime_image = os.environ.get("VLLM_RUNTIME_IMAGE", "")

        prompt = (
            "You are a supervisor agent that coordinates several specialist tools:\n"
            "- Configuration Specialist: preloaded model-car requirements, YAML-derived details, VRAM estimates, and "
            "  (when requested) applying optimized serving arguments back into the model-car configuration.\n"
            "- Accelerator Specialist: cluster accelerators, GPU/Spyre profiles, hardware details.\n"
            "- Decision Specialist: GO/NO-GO deployment decisions based on model requirements, accelerator capacity, "
            "  and serving arguments (e.g., tensor_parallel_size, max_model_len, executor backend).\n"
            "- QA Specialist: KServe deployment QA using deployment-yamls/ — applies registry secret and InferenceServices "
            "in namespace model-validation (sequential, smallest image first), monitors readiness and logs, "
            "and self-heals within bounded retries; reports QA_OK / QA_ERROR.\n\n"

            "A model-car configuration has already been processed by the host program. "
            "You can access its details only via your tools; never ask the user for YAML or file paths.\n"
            "- When you call the Accelerator Specialist, it will return a JSON blob that includes a field "
            "vllm_runtime_image indicating the vLLM runtime image to be used for deployment.\n"
            "If you get a GO decision from the Decision Specialist, you MUST use this exact image to call the QA Specialist.\n\n"
            "- If you need to call the QA Specialist (tool `analyze_qa_results`), you MUST:\n"
            "  - Set the `request` argument to a short natural-language instruction like\n"
            "    \"Run QA and summarize the validation results.\"\n"
            "  - Set the `runtime_image` argument to the exact value of \"vllm_runtime_image\" from the\n"
            f"    accelerator JSON. Do NOT invent or guess this value. But use `{runtime_image}` if its not empty.\n\n"

            "Environment and safety rules (CRITICAL):\n"
            "- After calling the Accelerator Specialist, if its report indicates ANY of the following:\n"
            "  * authentication problems (e.g. 'Unauthorized', 'Forbidden', 'cluster login failed'),\n"
            "  * connectivity issues (e.g. 'unable to reach cluster', TLS handshake errors), or\n"
            "  * that accelerators could not be inspected at all,\n"
            "  then you MUST treat the cluster as unavailable and deployment as NOT SAFE.\n"
            "- In that case:\n"
            "  * The final deployment verdict MUST be **NO-GO**, regardless of GPU capacity or model VRAM.\n"
            "  * You MUST NOT call the QA Specialist, because QA will also fail if the cluster is unreachable.\n"
            "  * When you call the Decision Specialist (if you choose to call it), clearly mention that the\n"
            "    accelerator step reported authentication / connectivity failure so it can also return NO-GO.\n\n"

            "Dynamic reasoning:\n"
            "- Read the user's request and decide which specialist tool(s) to call.\n"
            "- For generic triggers such as 'Start supervisor agent', you MUST perform a full deployment assessment:\n"
            "  1) Use the Configuration Specialist to summarise preloaded model requirements.\n"
            "  2) Use the Accelerator Specialist to inspect accelerators.\n"
            "     - If this step reports authentication or connectivity failures, immediately decide NO-GO as described\n"
            "       above and SKIP calling the QA Specialist.\n"
            "  3) Use the Decision Specialist to decide GO or NO-GO, taking into account GPU capacity, serving arguments,\n"
            "     and any environment health issues reported by the Accelerator Specialist.\n"
            "     - Authoritative deployability: the Decision Specialist persists `deployment_matrix.json` in the run info\n"
            "       directory (deterministic per-model rows). If every row has deployable=true and the Accelerator step\n"
            "       did not report authentication or connectivity failure, your ### Deployment Decision MUST state **GO**\n"
            "       and align reasoning with those rows — do not output NO-GO that contradicts the matrix.\n"
            "     - The Decision Specialist may also propose optimized serving arguments (usually as JSON) when current\n"
            "       arguments are suboptimal or unsafe (e.g., risk of OOM).\n"
            "  4) If the Decision Specialist explicitly recommends optimized serving arguments and the environment is\n"
            "     healthy, you SHOULD normally call the Configuration Specialist a second time and ask it to apply those\n"
            "     optimized arguments to the model-car YAML (for example, by passing the JSON blob to its tool that updates\n"
            "     serving_arguments). Do this BEFORE calling QA so that validation uses the optimized configuration.\n"
            "  5) If you are issuing a deployment verdict and the environment is healthy, you SHOULD normally call the\n"
            "     QA Specialist to run KServe deployment validation and include the results, unless the user explicitly says to skip QA.\n\n"

            "Tool usage guidelines:\n"
            "- You may call the Configuration Specialist more than once in a single run:\n"
            "  - First for a read-only summary of requirements and VRAM.\n"
            "  - Later (optionally) to apply optimized serving arguments suggested by the Decision Specialist.\n"
            "- When asking the Configuration Specialist to apply optimized arguments, clearly pass along the structured\n"
            "  data (for example: 'Apply these optimized serving arguments JSON: {...}') so it can update the model-car.\n"
            "- Never claim GO if any tool output mentions 'Unauthorized', 'Forbidden', 'cluster login failed', or a\n"
            "  similar message indicating missing credentials or broken cluster access.\n\n"

            "Output format:\n"
            "- Always respond with a single structured report with the following sections:\n"
            "  ### Configuration Summary\n"
            "  ### Accelerator Summary\n"
            "  ### Deployment Decision\n"
            "    - Clearly state GO or NO-GO and explain:\n"
            "      * GPU capacity vs model requirements, AND\n"
            "      * Serving-argument suitability, AND\n"
            "      * Environment / access health (e.g. auth failures, unreachable cluster).\n"
            "    - If optimized serving arguments were applied, explicitly mention that they were written back to the\n"
            "      model-car configuration. And provide a snippet of what optimization was applied.\n"
            "  ### QA Validation (even if you only say it was not run)\n"
            "- In each section, clearly state which facts came from which type of specialist.\n"
            "- Do not introduce yourself or explain that you are a supervisor.\n"
        )

        return create_agent(
            self.llm,
            tools=tools,
            system_prompt=prompt,
        )



    def _save_precomputed_requirements(self):
        """Save precomputed requirements to info/models_info.json."""
        if not self.precomputed_requirements:
            return
        
        if self.info_dir is not None:
            info_dir = self.info_dir
        else:
            start_paths: list[Path] = [Path(__file__).resolve()]
            if self.bootstrap_config_path:
                start_paths.append(self.bootstrap_config_path)
            repo_root = detect_repo_root(start_paths)
            info_dir = repo_root / "info"
        info_dir.mkdir(parents=True, exist_ok=True)
        
        models_info_path = info_dir / "models_info.json"
        
        try:
            with open(models_info_path, 'w') as f:
                json.dump(self.precomputed_requirements, f, indent=2)
            logger.info(f"Precomputed requirements saved to {models_info_path}")
        except Exception as e:
            logger.error(f"Failed to save precomputed requirements to {models_info_path}: {e}")

    @staticmethod
    def _extract_final_text(result: Dict[str, Any]) -> str:
        messages: List[Any] = result.get("messages", [])
        if not messages:
            for key in ("output", "output_text", "output_str"):
                text = result.get(key)
                if isinstance(text, str) and text.strip():
                    return text.strip()
            return ""

        final_message = messages[-1]
        content = getattr(final_message, "content", final_message)

        if isinstance(content, str):
            return content

        if isinstance(content, list):
            parts: List[str] = []
            for block in content:
                if isinstance(block, str):
                    parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    parts.append(block.get("text", ""))
            return "\n".join(part for part in parts if part).strip()

        return str(content)
