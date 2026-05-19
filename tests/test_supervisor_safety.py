"""Tests for supervisor auth-failure safety enforcement."""

from __future__ import annotations

import unittest

from runtimes_dep_agent.agent.llm_agent import _accelerator_indicates_auth_failure


class TestAcceleratorAuthFailureDetection(unittest.TestCase):
    """Verify _accelerator_indicates_auth_failure detects known auth/connectivity keywords."""

    # -- positive cases --------------------------------------------------

    def test_detects_unauthorized(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "Error: Unauthorized — please log in to the cluster."
            )
        )

    def test_detects_forbidden(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "403 Forbidden: user does not have access."
            )
        )

    def test_detects_cluster_login_failed(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "cluster login failed: token expired."
            )
        )

    def test_detects_unable_to_reach_cluster(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "unable to reach cluster at api.example.com:6443"
            )
        )

    def test_detects_tls_handshake(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "TLS handshake timeout connecting to api server."
            )
        )

    def test_detects_cannot_connect(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "cannot connect to the OpenShift API."
            )
        )

    def test_detects_connection_refused(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "dial tcp 10.0.0.1:6443: connection refused"
            )
        )

    def test_detects_authentication_failed(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure(
                "authentication failed for user admin."
            )
        )

    # -- case insensitivity -----------------------------------------------

    def test_case_insensitive_unauthorized(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure("UNAUTHORIZED access detected.")
        )

    def test_case_insensitive_forbidden(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure("FORBIDDEN: insufficient scope")
        )

    def test_case_insensitive_mixed(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure("Cluster Login Failed — retry later.")
        )

    def test_case_insensitive_tls(self) -> None:
        self.assertTrue(
            _accelerator_indicates_auth_failure("tLs HaNdShAkE error occurred")
        )

    # -- no false positives -----------------------------------------------

    def test_no_false_positive_normal_output(self) -> None:
        normal = (
            '{"gpu_available": true, "gpu_provider": "NVIDIA", '
            '"vllm_image": "registry.redhat.io/rhaiis/vllm-cuda-runtime-rhel9:latest"}'
        )
        self.assertFalse(_accelerator_indicates_auth_failure(normal))

    def test_no_false_positive_no_gpu(self) -> None:
        self.assertFalse(
            _accelerator_indicates_auth_failure(
                "No GPU available in the cluster. Provider: NONE"
            )
        )

    def test_no_false_positive_empty_string(self) -> None:
        self.assertFalse(_accelerator_indicates_auth_failure(""))

    def test_no_false_positive_authorized_keyword(self) -> None:
        """'authorized' should not match (we look for 'unauthorized')."""
        self.assertFalse(
            _accelerator_indicates_auth_failure(
                "User is authorized to access the cluster."
            )
        )

    def test_no_false_positive_success_message(self) -> None:
        self.assertFalse(
            _accelerator_indicates_auth_failure(
                "GPU information saved to /tmp/gpu_info.json. GPU Provider: NVIDIA."
            )
        )

    # -- keyword embedded in longer text ----------------------------------

    def test_detects_keyword_embedded_in_paragraph(self) -> None:
        text = (
            "Accelerator Validation Result:\n"
            "Status: Error\n"
            "The cluster returned 401 Unauthorized when querying node resources.\n"
            "Recommendation: Re-authenticate with oc login."
        )
        self.assertTrue(_accelerator_indicates_auth_failure(text))


if __name__ == "__main__":
    unittest.main()
