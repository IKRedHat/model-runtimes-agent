"""Unit tests for KServe QA helpers (no cluster required)."""

from __future__ import annotations

import json
import unittest

from runtimes_dep_agent.qa_kserve.heuristics import classify_pod_json, logs_hint_oom
from runtimes_dep_agent.qa_kserve.render import (
    format_args_block,
    halve_max_model_len_args,
    normalize_dockerconfig_b64,
    render_inference_service,
    sanitize_k8s_name,
    validate_yaml_document,
)
from runtimes_dep_agent.qa_kserve import oc_cli


class TestSanitizeName(unittest.TestCase):
    def test_sanitizes(self) -> None:
        self.assertEqual(sanitize_k8s_name("Model_1_Foo!"), "model-1-foo")
        self.assertTrue(sanitize_k8s_name("1abc").startswith("m-"))


class TestDockerconfigB64(unittest.TestCase):
    def test_raw_json_encoded(self) -> None:
        raw = '{"auths":{}}'
        b64 = normalize_dockerconfig_b64(raw)
        self.assertNotIn("{", b64)
        self.assertEqual(normalize_dockerconfig_b64("  dGVzdA==  "), "dGVzdA==")  # already b64


class TestClassifyPods(unittest.TestCase):
    def test_classify_oom(self) -> None:
        pod = {
            "items": [
                {
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "kserve-container",
                                "state": {
                                    "terminated": {"reason": "OOMKilled", "exitCode": 137}
                                },
                            }
                        ]
                    }
                }
            ]
        }
        kind, _ = classify_pod_json(json.dumps(pod))
        self.assertEqual(kind, "oom")

    def test_classify_image_pull(self) -> None:
        pod = {
            "items": [
                {
                    "status": {
                        "containerStatuses": [
                            {
                                "name": "kserve-container",
                                "state": {
                                    "waiting": {"reason": "ImagePullBackOff"}
                                },
                            }
                        ]
                    }
                }
            ]
        }
        kind, _ = classify_pod_json(json.dumps(pod))
        self.assertEqual(kind, "image_pull")


class TestRenderInference(unittest.TestCase):
    def test_roundtrip_yaml(self) -> None:
        tmpl = """apiVersion: serving.kserve.io/v1beta1
kind: InferenceService
metadata:
  name: __ISVC_NAME__
  namespace: model-validation
  annotations:
    deployment.qa/vllm-runtime-image: "__VLLM_RUNTIME_IMAGE__"
spec:
  predictor:
    minReplicas: 1
    maxReplicas: 1
__IMAGE_PULL_SECRETS_BLOCK__
    model:
      runtime: __SERVING_RUNTIME_NAME__
      modelFormat:
        name: __MODEL_FORMAT__
      storageUri: "__MODEL_IMAGE__"
__ARGS_BLOCK__
    resources:
      requests:
        cpu: "__CPU_REQUEST__"
        memory: "__MEMORY_REQUEST__"
__GPU_REQUESTS_LINE__
      limits:
        cpu: "__CPU_LIMIT__"
        memory: "__MEMORY_LIMIT__"
"""
        body = render_inference_service(
            template_text=tmpl,
            isvc_name="m-x",
            model_image="oci://reg/ns/img:1",
            vllm_runtime_image="quay.io/vllm:1",
            serving_runtime_name="vllm-runtime",
            model_format="huggingface",
            args=["--max-model-len=2048"],
            oci_secret_name="pull-secret",
            cpu_request="2",
            memory_request="8Gi",
            cpu_limit="8",
            memory_limit="32Gi",
            gpu_count=1,
        )
        doc = validate_yaml_document(body)
        self.assertEqual(doc["kind"], "InferenceService")
        self.assertEqual(doc["metadata"]["name"], "m-x")


class TestArgsHelpers(unittest.TestCase):
    def test_halve_max_model_len(self) -> None:
        args = ["--max-model-len=4096", "--tensor-parallel-size=1"]
        out = halve_max_model_len_args(args)
        self.assertIn("--max-model-len=2048", out)

    def test_format_args_block(self) -> None:
        block = format_args_block(['--foo="bar"'])
        self.assertIn("--foo=", block)


class TestLogsHint(unittest.TestCase):
    def test_oom_hint(self) -> None:
        self.assertTrue(logs_hint_oom("CUDA out of memory"))


class TestOcAllowlist(unittest.TestCase):
    def test_rejects_unknown(self) -> None:
        with self.assertRaises(ValueError):
            oc_cli.run_oc(["exec", "pod", "x"])


if __name__ == "__main__":
    unittest.main()
