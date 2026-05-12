"""Validators for accelerator compatibility.

Provides validation for:
1. CUDA compatibility
2. ROCm compatibility
3. vLLM-Spyre-x86 compatibility
4. GPU memory and capacity requirements
"""

import os
import subprocess
import re
import json
from pathlib import Path


def check_gpu_availability() -> tuple[bool, str]:
    """
    Check if the GPU is available by running OpenShift command to check for GPU nodes.
    
    Returns:
        tuple: (gpu_status: bool, gpu_provider: str)
        - gpu_status: True if GPU is available, False otherwise
        - gpu_provider: "NVIDIA", "AMD", or "NONE"
    """
    try:
        # Run the oc command to get GPU information from nodes
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "custom-columns=NAME:.metadata.name,GPUs:.status.allocatable", "--no-headers"],
            capture_output=True,
            text=True,
            timeout=30
        )

        available_gpus = {
            "nvidia.com/gpu": "NVIDIA",
            "amd.com/gpu": "AMD",
            "ibm.com/spyre_pf": "SPYRE_x86",
            "ibm.com/spyre_vf": "SPYRE_s390x",
            "habana.ai/gaudi": "INTEL",
        }

        if result.returncode == 0:
            output = result.stdout
            if "nvidia.com/gpu" in output:
                return True, available_gpus["nvidia.com/gpu"]
            elif "amd.com/gpu" in output:
                return True, available_gpus["amd.com/gpu"]
            elif "ibm.com/spyre_pf" in output:
                return True, available_gpus["ibm.com/spyre_pf"]
            elif "ibm.com/spyre_vf" in output:
                return True, available_gpus["ibm.com/spyre_vf"]
            elif "habana.ai/gaudi" in output:
                return True, available_gpus["habana.ai/gaudi"]
            else:
                return False, "NONE"
        else:
            return False, "NONE"
            
    except subprocess.TimeoutExpired:
        return False, "NONE"
    except FileNotFoundError:
        return False, "NONE"
    except Exception:
        return False, "NONE"


def get_gpu_info(info_dir: Path | str | None = None):
    """
    Get detailed GPU information based on the GPU provider and save to gpu_info.txt file.
    When info_dir is provided, the file is written there; otherwise the info folder
    in the project root is used (overwrites existing file).

    Args:
        info_dir: Optional per-run info directory; when None, uses project root / "info".

    Returns:
        str: Absolute path to the created gpu_info.txt file
    """
    if info_dir is not None:
        info_dir = Path(info_dir)
        info_dir.mkdir(parents=True, exist_ok=True)
        file_path = info_dir / "gpu_info.txt"
    else:
        # Find project root (where config-yaml folder is located)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != os.path.dirname(current_dir):
            if os.path.exists(os.path.join(current_dir, "config-yaml")):
                break
            current_dir = os.path.dirname(current_dir)
        info_dir = os.path.join(current_dir, "info")
        os.makedirs(info_dir, exist_ok=True)
        file_path = os.path.join(info_dir, "gpu_info.txt")
    
    try:
        # Get GPU availability status
        gpu_status, gpu_provider = check_gpu_availability()
        
        if not gpu_status:
            info_content = "No GPU available in the cluster\n"
        else:
            # Get detailed GPU information based on provider
            if gpu_provider == "NVIDIA":
                info_content = get_nvidia_gpu_details()
            elif gpu_provider == "AMD":
                info_content = get_amd_gpu_details()
            elif gpu_provider.startswith("SPYRE"):
                info_content = get_spyre_gpu_details()
            elif gpu_provider == "INTEL":
                info_content = get_intel_gpu_details()
            else:
                info_content = f"Unknown GPU provider: {gpu_provider}\n"
        
        # Overwrite existing file
        with open(file_path, "w") as f:
            f.write(info_content)
        
        return file_path
        
    except Exception as e:
        error_content = f"Error getting GPU info: {str(e)}\n"
        # Overwrite existing file with error
        with open(file_path, "w") as f:
            f.write(error_content)
        return file_path

def check_cluster_login():
    """
    Check if the cluster is logged in by running 'oc cluster-info' command.
    
    Returns:
        str: "true cluster logged in" if successful, "failed" if not
    """
    try:
        result = subprocess.run(
            ["oc", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode == 0:
            return "Cluster is logged in"
        else:
            return "Login to your cluster to continue"

    except subprocess.TimeoutExpired:
        return "failed"
    except FileNotFoundError:
        return "failed"
    except Exception:
        return "failed"


def _get_cloud_provider():
    """
    Get cloud provider from oc cluster-info command.
    
    Returns:
        str: Cloud provider name (AWS, Azure, GCP, IBM, etc.) or "Unknown"
    """
    try:
        result = subprocess.run(
            ["oc", "cluster-info"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            output = result.stdout.lower()
            
            # Check for  cloud providers 
            if "amazonaws.com" in output or "aws" in output:
                return "AWS"
            elif "azure.com" in output or "azure" in output:
                return "Azure"
            elif "googleapis.com" in output or "gcp" in output or "google" in output:
                return "GCP"
            elif "ibm.com" in output or "ibm" in output:
                return "IBM"
            else:
                return "Unknown"
        else:
            return "Unknown"
            
    except Exception:
        return "Unknown"


def _mem_from_product(product: str) -> float | None:
    if not product:
        return None
    m = re.search(r"(\d+(?:\.\d+)?)\s*gb", product, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None


def _convert_to_gb(value):
    """
    Convert Kubernetes memory/storage values to GB.
    
    Args:
        value: Memory or storage value (e.g., "16Gi", "1000Mi", "5000000000")
        
    Returns:
        str: Value in GB format
    """
    if not value or value == "Unknown":
        return "Unknown"
    
    try:
        # Handle different units
        if value.endswith('Gi'):
            # Already in GiB, convert to GB (1 GiB = 1.074 GB)
            num = float(value[:-2])
            return f"{num * 1.074:.1f}"
        elif value.endswith('Mi'):
            # Convert MiB to GB
            num = float(value[:-2])
            return f"{num / 1024 * 1.074:.1f}"
        elif value.endswith('Ki'):
            # Convert KiB to GB
            num = float(value[:-2])
            return f"{num / (1024 * 1024) * 1.074:.1f}"
        elif value.endswith('G'):
            # Already in GB
            return value[:-1]
        elif value.endswith('M'):
            # Convert MB to GB
            num = float(value[:-1])
            return f"{num / 1000:.1f}"
        elif value.endswith('K'):
            # Convert KB to GB
            num = float(value[:-1])
            return f"{num / (1000 * 1000):.1f}"
        else:
            # Assume it's in bytes, convert to GB
            num = float(value)
            return f"{num / (1000 * 1000 * 1000):.1f}"
    except (ValueError, TypeError):
        return "Unknown"


def get_nvidia_gpu_details():
    """Get detailed NVIDIA GPU information."""
    try:
        # Get nodes with NVIDIA GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            if "nvidia.com/gpu" in allocatable:
                node_name = node.get("metadata", {}).get("name", "Unknown")
                labels = node.get("metadata", {}).get("labels", {})
                gpu_count = allocatable.get("nvidia.com/gpu", "0")
                product = labels.get("nvidia.com/gpu.product", "Unknown")
                cpumemory = allocatable.get("memory", "Unknown")
                storage = allocatable.get("ephemeral-storage", "Unknown")

                per_gpu_mem_gb = None
                gpu_mem_mib = labels.get("nvidia.com/gpu.memory")

                if gpu_mem_mib is not None:
                    try:
                        # label is in MiB, convert to GiB
                        per_gpu_mem_gb = round(int(gpu_mem_mib) / 1024, 2)
                    except ValueError:
                        per_gpu_mem_gb = None
                else:
                    per_gpu_mem_gb = _mem_from_product(product)
                per_gpu_mem_str = f"{per_gpu_mem_gb} GB" if per_gpu_mem_gb is not None else "Unknown"
                
                # Get instance type from labels
                instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                
                # Get cloud provider from cluster info
                cloud_provider = _get_cloud_provider()
                
                # Convert memory to GB
                memory_gb = _convert_to_gb(cpumemory)
                storage_gb = _convert_to_gb(storage)
                
                gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                gpu_info.append(f"• Instance Type: {instance_type}")
                gpu_info.append(f"• GPU Provider: NVIDIA")
                gpu_info.append(f"• GPU Product: {product}")
                gpu_info.append(f"• Per-GPU Memory: {per_gpu_mem_str} GB")
                gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                gpu_info.append(f"• Node RAM: {memory_gb} GB")
                gpu_info.append(f"• Node Storage: {storage_gb} GB")
                gpu_info.append(f"• Node Name: {node_name}")
                gpu_info.append("")  # Empty line between nodes
        
        return "\n".join(gpu_info) if gpu_info else "No NVIDIA GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting NVIDIA GPU details: {str(e)}\n"


def get_amd_gpu_details():
    """Get detailed AMD GPU information."""
    try:
        # Get nodes with AMD GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            if "amd.com/gpu" in allocatable:
                node_name = node.get("metadata", {}).get("name", "Unknown")
                gpu_count = allocatable.get("amd.com/gpu", "0")
                memory = allocatable.get("memory", "Unknown")
                storage = allocatable.get("ephemeral-storage", "Unknown")
                
                # Get instance type from labels
                labels = node.get("metadata", {}).get("labels", {})
                instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                
                # Get cloud provider from cluster info
                cloud_provider = _get_cloud_provider()
                
                # Try to determine AMD GPU model from instance type
                gpu_model = "Unknown"
                if "MI300X" in instance_type:
                    gpu_model = "MI300X"
                elif "MI250" in instance_type:
                    gpu_model = "MI250"
                elif "MI100" in instance_type:
                    gpu_model = "MI100"

                # Map known AMD accelerators to their per-GPU VRAM (in GB)
                amd_vram_lookup = {
                    "MI300X": 192,
                    "MI250": 128,
                    "MI210": 64,
                    "MI100": 32,
                }
                per_gpu_mem = amd_vram_lookup.get(gpu_model)
                
                # Convert memory to GB
                memory_gb = _convert_to_gb(memory)
                storage_gb = _convert_to_gb(storage)
                
                gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                gpu_info.append(f"• Instance Type: {instance_type}")
                gpu_info.append(f"• GPU Provider: AMD ({gpu_model})")
                if per_gpu_mem is not None:
                    gpu_info.append(f"• Per-GPU Memory: {per_gpu_mem} GB")
                gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                gpu_info.append(f"• Memory: {memory_gb} GB")
                gpu_info.append(f"• Storage: {storage_gb} GB")
                gpu_info.append(f"• Node Name: {node_name}")
                gpu_info.append("")  # Empty line between nodes
        
        return "\n".join(gpu_info) if gpu_info else "No AMD GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting AMD GPU details: {str(e)}\n"
    
def get_spyre_gpu_details():
    """Get detailed SPYRE GPU information."""
    try:
        # Get nodes with SPYRE GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            for spyre_key, spyre_label in [("ibm.com/spyre_pf", "SPYRE_x86"), ("ibm.com/spyre_vf", "SPYRE_s390x")]:
                if spyre_key in allocatable:
                    node_name = node.get("metadata", {}).get("name", "Unknown")
                    gpu_count = allocatable.get(spyre_key, "0")
                    memory = allocatable.get("memory", "Unknown")
                    storage = allocatable.get("ephemeral-storage", "Unknown")
                    
                    # Get instance type from labels
                    labels = node.get("metadata", {}).get("labels", {})
                    instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                    
                    # Get cloud provider from cluster info
                    cloud_provider = _get_cloud_provider()
                    
                    # Convert memory to GB
                    memory_gb = _convert_to_gb(memory)
                    storage_gb = _convert_to_gb(storage)
                    
                    gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                    gpu_info.append(f"• Instance Type: {instance_type}")
                    gpu_info.append(f"• GPU Provider: SPYRE")
                    gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                    gpu_info.append(f"• Memory: {memory_gb} GB")
                    gpu_info.append(f"• Storage: {storage_gb} GB")
                    gpu_info.append(f"• Node Name: {node_name}")
                    gpu_info.append("")  # Empty line between nodes
            
        return "\n".join(gpu_info) if gpu_info else "No SPYRE GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting SPYRE GPU details: {str(e)}\n"
    
def get_intel_gpu_details():
    """Get detailed Intel GPU information."""
    try:
        # Get nodes with Intel GPU
        result = subprocess.run(
            ["oc", "get", "nodes", "-o", "json"],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode != 0:
            return "Failed to get node information\n"
        nodes_data = json.loads(result.stdout)
        
        gpu_info = []
        for node in nodes_data.get("items", []):
            allocatable = node.get("status", {}).get("allocatable", {})
            if "habana.ai/gaudi" in allocatable:
                node_name = node.get("metadata", {}).get("name", "Unknown")
                gpu_count = allocatable.get("habana.ai/gaudi", "0")
                memory = allocatable.get("memory", "Unknown")
                storage = allocatable.get("ephemeral-storage", "Unknown")
                
                # Get instance type from labels
                labels = node.get("metadata", {}).get("labels", {})
                instance_type = labels.get("node.kubernetes.io/instance-type", "Unknown")
                
                # Get cloud provider from cluster info
                cloud_provider = _get_cloud_provider()
                
                # Convert memory to GB
                memory_gb = _convert_to_gb(memory)
                storage_gb = _convert_to_gb(storage)
                
                gpu_info.append(f"• Cloud Provider: {cloud_provider}")
                gpu_info.append(f"• Instance Type: {instance_type}")
                gpu_info.append(f"• GPU Provider: INTEL")
                gpu_info.append(f"• Allocatable GPUs: {gpu_count}")
                gpu_info.append(f"• Memory: {memory_gb} GB")
                gpu_info.append(f"• Storage: {storage_gb} GB")
                gpu_info.append(f"• Node Name: {node_name}")
                gpu_info.append("")  # Empty line between nodes
        
        return "\n".join(gpu_info) if gpu_info else "No INTEL GPU nodes found\n"
        
    except Exception as e:
        return f"Error getting INTEL GPU details: {str(e)}\n"
    
def get_vllm_runtime_image_from_template(
        gpu_provider: str
) -> str:
    """
    Return the vLLM runtime container image from an OpenShift Template based on the GPU provider.

    Args:
        gpu_provider (str): The GPU provider ("NVIDIA", "AMD", "SPYRE_x86", "SPYRE_s390x", "INTEL", "NONE").

    It expects a Template like `vllm-cuda-runtime-template` that contains a
    single ServingRuntime in `objects[0]` and reads:

      objects[0].spec.containers[0].image

    Raises:
        RuntimeError if the template or container/image cannot be resolved.
    """
    available_templates = {
        "NVIDIA": "vllm-cuda-runtime-template",
        "AMD": "vllm-rocm-runtime-template",
        "SPYRE_x86": "vllm-spyre-x86-runtime-template",
        "SPYRE_s390x": "vllm-spyre-s390x-runtime-template",
        "INTEL": "vllm-gaudi-runtime-template",
        "NONE": "vllm-cuda-runtime-template",
    }
    if gpu_provider not in available_templates:
        raise RuntimeError(f"No vLLM runtime template for GPU provider: {gpu_provider}")
    cmd = [
        "oc",
        "get",
        "template",
        available_templates[gpu_provider],
        "-n",
        "redhat-ods-applications",
        "-o",
        "json",
    ]

    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to get template vllm-cuda-runtime-template in redhat-ods-applications: {e.stderr or e.stdout}"
        ) from e

    try:
        data = json.loads(completed.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(
            f"Failed to parse JSON from oc output for template vllm-cuda-runtime-template: {e}"
        ) from e

    objects = data.get("objects") or []
    if not objects:
        raise RuntimeError(
            f"Template vllm-cuda-runtime-template in redhat-ods-applications has no objects field."
        )

    sr = objects[0]
    spec = sr.get("spec") or {}
    containers = spec.get("containers") or []
    if not containers:
        raise RuntimeError(
            f"ServingRuntime in template vllm-cuda-runtime-template has no spec.containers."
        )

    image = containers[0].get("image")
    if not image:
        raise RuntimeError(
            f"First container in template vllm-cuda-runtime-template has no image field."
        )

    return image


_VLLM_TEMPLATE_KEYS = frozenset(
    {"NVIDIA", "AMD", "SPYRE_x86", "SPYRE_s390x", "INTEL", "NONE"},
)


def normalize_gpu_provider_for_vllm_template(gpu_provider: str | None) -> str:
    """
    Map QA / UI strings to keys accepted by :func:`get_vllm_runtime_image_from_template`.

    CPU, empty, and unrecognized values map to ``NONE`` (RHOAI ``vllm-cuda-runtime-template``).
    """
    if gpu_provider is None:
        return "NONE"
    p = str(gpu_provider).strip()
    if not p:
        return "NONE"
    if p in _VLLM_TEMPLATE_KEYS:
        return p
    pu = p.upper()
    if pu in ("CPU", "NONE"):
        return "NONE"
    if pu == "NVIDIA":
        return "NVIDIA"
    if pu == "AMD":
        return "AMD"
    if pu == "INTEL":
        return "INTEL"
    if "SPYRE" in pu:
        if "S390" in pu or "390" in pu:
            return "SPYRE_s390x"
        return "SPYRE_x86"
    return "NONE"


# Test the functions
if __name__ == "__main__":
    gpu_status, gpu_provider = check_gpu_availability()
    print(f"GPU Status: {gpu_status}")
    print(f"GPU Provider: {gpu_provider}")
    
    if gpu_status:
        file_path = get_gpu_info()
        print(f"GPU information saved to: {file_path}")
    
