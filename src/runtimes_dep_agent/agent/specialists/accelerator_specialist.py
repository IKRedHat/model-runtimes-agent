"""Accelerator specialist agent for GPU validation and compatibility checking."""

from __future__ import annotations

import json
import os
from typing import Callable

from langchain.agents import create_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import tool

from pathlib import Path

from . import SpecialistSpec
from ...validators.accelerator_validator import (
    check_cluster_login,
    check_gpu_availability,
    get_gpu_info,
    get_vllm_runtime_image_from_template,
    normalize_gpu_provider_for_vllm_template,
)


def build_accelerator_specialist(
    llm: BaseChatModel,
    extract_text: Callable[[dict], str],
    precomputed_requirements: dict | None = None,
    info_dir: Path | None = None,
) -> SpecialistSpec:
    """Return the accelerator specialist agent and the supervisor-facing tool."""

    @tool
    def check_cluster_authentication() -> str:
        """Check if the user is logged into the OpenShift cluster."""
        return check_cluster_login()

    @tool
    def check_gpu_status() -> str:
        """Check GPU availability in the cluster.
        
        Returns:
            str: GPU status and provider information (NVIDIA, AMD, or NONE).
        """
        gpu_status, gpu_provider = check_gpu_availability()
        if gpu_status:
            return f"GPU is available. Provider: {gpu_provider}"
        else:
            return f"No GPU available in the cluster. Provider: {gpu_provider}"

    @tool
    def get_detailed_gpu_information() -> str:
        """Get detailed GPU information from the cluster and save to file.
        
        Returns:
            str: Path to the GPU info file and summary of GPU details.
        """
        # First check cluster login
        login_status = check_cluster_login()
        if "failed" in login_status.lower() or "login" in login_status.lower():
            return f"Error: {login_status}. Please login to the cluster first."
        
        file_path = get_gpu_info(info_dir)
        gpu_status, gpu_provider = check_gpu_availability()
        
        if gpu_status:
            return f"GPU information saved to {file_path}. GPU Provider: {gpu_provider}. Check the file for detailed information."
        else:
            return f"GPU information saved to {file_path}. No GPU available in the cluster."

    @tool
    def get_accelerator_metadata_json() -> str:
        """
        Return accelerator metadata as JSON.
        Example response:
            {
            "gpu_available": true,
            "gpu_provider": "NVIDIA",
            "vllm_image": "registry.redhat.io/rhaiis/vllm-cuda-runtime-rhel9:latest"
            }
        """
        gpu_status, gpu_provider = check_gpu_availability()
        override_image = os.environ.get("VLLM_RUNTIME_IMAGE")
        vllm_image = override_image or get_vllm_runtime_image_from_template(
            normalize_gpu_provider_for_vllm_template(gpu_provider)
        )
        
        metadata = {
            "gpu_available": gpu_status,
            "gpu_provider": gpu_provider,
            "vllm_image": vllm_image
        }
        
        return json.dumps(metadata, indent=2)

    @tool
    def validate_accelerator_compatibility(request: str) -> str:
        """Validate accelerator compatibility for models.
        
        Args:
            request: User request about accelerator validation or compatibility.
            
        Returns:
            str: Validation results and compatibility information.
        """
        # Check cluster login first
        login_status = check_cluster_login()
        if "failed" in login_status.lower() or "login" in login_status.lower():
            return f"Error: {login_status}. Cannot validate accelerators without cluster access."
        
        # Get GPU information
        gpu_status, gpu_provider = check_gpu_availability()
        
        if not gpu_status:
            return (
                "Accelerator Validation Result:\n"
                "Status: No GPU available in the cluster\n"
                "Provider: NONE\n"
                "Recommendation: Ensure GPU nodes are available in the cluster."
            )
        
        # Get detailed info
        file_path = get_gpu_info(info_dir)
        
        result = (
            f"Accelerator Validation Result:\n"
            f"Status: GPU available\n"
            f"Provider: {gpu_provider}\n"
            f"Detailed information saved to: {file_path}\n"
        )
        
        # Add compatibility notes based on provider
        if gpu_provider == "NVIDIA":
            result += "\nCompatibility Notes:\n"
            result += "- CUDA-compatible models are supported\n"
            result += "- vLLM compatibility available\n"
        elif gpu_provider == "AMD":
            result += "\nCompatibility Notes:\n"
            result += "- ROCm-compatible models are supported\n"
            result += "- Check model requirements for AMD GPU compatibility\n"
        
        return result

    prompt = (
        "You are an accelerator and GPU compatibility specialist. "
        "Begin with a short checklist using [ ] / [x] to show the steps you will take "
        "(e.g., load cached requirements, check cluster authentication, query GPU status, fetch detailed info). "
        "Mark steps complete as you invoke the tools. "
        "Use the provided tools to check GPU availability, validate accelerator compatibility, "
        "and provide detailed GPU information from OpenShift clusters and the GPU size. "
        "Never ask the user for model requirements; rely on the cached JSON. "
        "Provide clear, structured responses with validation results and recommendations."
        "You must always return a machine readable json output back to Supervisor by using the tool "
        "get_accelerator_metadata_json() as a final step. And provide this json output to the Supervisor."

    )

    agent = create_agent(
        llm,
        tools=[
            check_cluster_authentication,
            check_gpu_status,
            get_detailed_gpu_information,
            validate_accelerator_compatibility,
            get_accelerator_metadata_json
        ],
        system_prompt=prompt,
    )

    @tool
    def analyze_accelerator(request: str) -> str:
        """Delegate accelerator and GPU validation requests to the accelerator specialist.
        Always return the JSON output from get_accelerator_metadata_json() to the Supervisor.
        
        """
        result = agent.invoke({"messages": [{"role": "user", "content": request}]})
        return extract_text(result)

    analyze_accelerator.name = "analyze_accelerator"

    return SpecialistSpec(
        name="accelerator_specialist",
        agent=agent,
        tool=analyze_accelerator,
    )


__all__ = ["build_accelerator_specialist"]
