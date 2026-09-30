"""Roll the KServe predictor to the current champion from inside the cluster (ADR-0006).

The InferenceService's storageUri is `models:/<model>@champion`, resolved when a
predictor pod starts, so a promotion needs new pods. Setting the pod annotation
`fraud-detection/model-version` on the InferenceService makes KServe roll the
predictor Deployment (a plain `rollout restart` is reverted by KServe).

`scripts/promote.py` does this with kubectl from a laptop; the retraining Job
has no kubectl, so it talks to the Kubernetes API directly with its
ServiceAccount token (Role `retrainer` in deploy/base/retrain/). Standard
library only.
"""

from __future__ import annotations

import json
import os
import ssl
import time
import urllib.request
from pathlib import Path
from typing import Any

ANNOTATION = "fraud-detection/model-version"
SA_DIR = Path("/var/run/secrets/kubernetes.io/serviceaccount")


class RolloutError(RuntimeError):
    """The predictor did not roll to the requested version in time."""


def in_cluster() -> bool:
    return "KUBERNETES_SERVICE_HOST" in os.environ and (SA_DIR / "token").exists()


class KubeApi:
    """Just enough of the Kubernetes API: GET and JSON merge-patch."""

    def __init__(self, base: str | None = None, token: str | None = None) -> None:
        host = os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc")
        port = os.environ.get("KUBERNETES_SERVICE_PORT", "443")
        self.base = base or f"https://{host}:{port}"
        self.token = token if token is not None else (SA_DIR / "token").read_text().strip()
        ca = SA_DIR / "ca.crt"
        self.ssl = ssl.create_default_context(cafile=str(ca)) if ca.exists() else None

    def _call(self, method: str, path: str, body: Any = None, content_type: str = "") -> Any:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        req.add_header("Accept", "application/json")
        if content_type:
            req.add_header("Content-Type", content_type)
        with urllib.request.urlopen(req, timeout=30, context=self.ssl) as resp:
            return json.load(resp)

    def get(self, path: str) -> Any:
        return self._call("GET", path)

    def merge_patch(self, path: str, body: Any) -> Any:
        return self._call("PATCH", path, body, "application/merge-patch+json")


def rollout(
    api: KubeApi,
    namespace: str,
    isvc: str,
    version: str,
    *,
    timeout_s: float = 600,
    poll_s: float = 5,
) -> None:
    """Annotate the InferenceService with `version` and wait for its predictor to roll."""
    isvc_path = f"/apis/serving.kserve.io/v1beta1/namespaces/{namespace}/inferenceservices/{isvc}"
    deploy_path = f"/apis/apps/v1/namespaces/{namespace}/deployments/{isvc}-predictor"
    api.merge_patch(isvc_path, {"spec": {"predictor": {"annotations": {ANNOTATION: version}}}})

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if _rolled_out(api.get(deploy_path), version):
            return
        time.sleep(poll_s)
    raise RolloutError(f"{namespace}/{isvc}-predictor did not roll to v{version} in {timeout_s}s")


def _rolled_out(deployment: dict[str, Any], version: str) -> bool:
    """KServe copied the annotation into the pod template and every replica runs it."""
    template = deployment["spec"]["template"]["metadata"].get("annotations", {})
    if template.get(ANNOTATION) != version:
        return False
    status = deployment.get("status", {})
    want = deployment["spec"].get("replicas", 1)
    return bool(
        status.get("observedGeneration", 0) >= deployment["metadata"]["generation"]
        and status.get("updatedReplicas", 0) == want
        and status.get("availableReplicas", 0) == want
        and status.get("replicas", 0) == want  # old pods gone
    )
