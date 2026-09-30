"""KServeScorer: the champion model served by KServe, over the V2 inference protocol (Phase 4).

The request is the `ml.features` vector as one named FP64 input per feature
(ADR-0013); the response's first output is predict_proba, so the fraud
probability is column 1. MLServer reports the MLflow registry version that
the storage initializer downloaded (ADR-0006) in `model_version`.

Before a served version's scores are used, its `feature_version` tag is
looked up in MLflow once and compared with this build's `FEATURE_VERSION`.
A mismatch, or a response with no version, raises `IncompatibleModelError`
and the request is scored by rules instead. Lookup failures are not cached,
so a brief MLflow outage only costs fallbacks until it recovers.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from typing import Any

import httpx2 as httpx
import structlog

from ml.features import FEATURE_NAMES, FEATURE_VERSION
from payments_api import metrics
from payments_api.scoring.base import ScoreResult
from payments_api.scoring.errors import IncompatibleModelError

log = structlog.get_logger(__name__)

# Returns the `feature_version` tag of a registry version (None if untagged). Blocking.
FeatureVersionLookup = Callable[[str], str | None]


class KServeTimeoutError(RuntimeError):
    """The predictor did not answer within the timeout."""


class KServeHTTPError(RuntimeError):
    """The predictor could not be reached, or answered with a non-2xx status."""


class KServeResponseError(RuntimeError):
    """The predictor answered 2xx with a body that is not the expected V2 response."""


class ModelVersionCheckError(RuntimeError):
    """The served version's feature_version tag could not be read from MLflow."""


def mlflow_feature_version_lookup(
    tracking_uri: str | None, model_name: str
) -> FeatureVersionLookup:
    """The production lookup: the tag on the registry version.

    MLflow is imported here, when the scorer is built at startup, not inside the
    first lookup: a cold `import mlflow` takes longer than the lookup timeout,
    so the first scored payment would otherwise fall back to rules.
    """
    from mlflow import MlflowClient

    client = MlflowClient(tracking_uri)

    def lookup(version: str) -> str | None:
        tags: dict[str, str] = client.get_model_version(model_name, version).tags
        return tags.get("feature_version")

    return lookup


class KServeScorer:
    name = "kserve"

    def __init__(
        self,
        *,
        url: str,
        model_name: str,
        timeout_s: float,
        lookup: FeatureVersionLookup,
        lookup_timeout_s: float = 2.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.url = url
        self.ready_url = url.removesuffix("/infer") + "/ready"
        self.model_name = model_name
        self.timeout_s = timeout_s
        self.lookup_timeout_s = lookup_timeout_s
        self._lookup = lookup
        self._transport = transport
        self._client: httpx.AsyncClient | None = None
        self._verified: set[str] = set()
        self._rejected: dict[str, str] = {}  # version -> reason
        self._pending: dict[str, asyncio.Task[str | None]] = {}
        self._current: str | None = None

    @property
    def model_version(self) -> str:
        return f"{self.model_name}/{self._current}" if self._current else "unknown"

    async def start(self) -> None:
        self._client = httpx.AsyncClient(timeout=self.timeout_s, transport=self._transport)
        if not await self.ready():
            log.warning("kserve_not_ready_at_startup", url=self.ready_url)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def ready(self) -> bool:
        if self._client is None:
            return False
        try:
            resp = await self._client.get(self.ready_url)
        except httpx.HTTPError:
            return False
        return resp.status_code == 200

    async def score(self, features: Mapping[str, float]) -> ScoreResult:
        if self._client is None:
            raise KServeHTTPError("scorer not started")
        payload = {
            "inputs": [
                {"name": n, "datatype": "FP64", "shape": [1, 1], "data": [float(features[n])]}
                for n in FEATURE_NAMES
            ]
        }
        try:
            resp = await self._client.post(self.url, json=payload)
        except httpx.TimeoutException as exc:
            raise KServeTimeoutError(f"no answer within {self.timeout_s}s") from exc
        except httpx.HTTPError as exc:
            raise KServeHTTPError(repr(exc)) from exc
        if resp.status_code != 200:
            raise KServeHTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")

        score, version = _parse(resp)
        await self._check_version(version)
        if version != self._current:
            previous, self._current = self.model_version, version
            metrics.set_model_version(self.name, self.model_version)
            log.info("model_version_changed", model_version=self.model_version, previous=previous)
        return ScoreResult(
            score=score, scorer=self.name, model_version=f"{self.model_name}/{version}"
        )

    async def _check_version(self, version: str | None) -> None:
        if not version:
            raise IncompatibleModelError("predictor did not report a model_version")
        if version in self._verified:
            return
        if version in self._rejected:
            raise IncompatibleModelError(self._rejected[version])

        # One lookup per version, shared by concurrent requests; bounded so a slow
        # MLflow costs a fallback, not a stuck request.
        task = self._pending.get(version)
        if task is None:
            task = asyncio.create_task(asyncio.to_thread(self._lookup, version))
            self._pending[version] = task
            task.add_done_callback(lambda _: self._pending.pop(version, None))
        try:
            trained_on = await asyncio.wait_for(asyncio.shield(task), self.lookup_timeout_s)
        except Exception as exc:
            raise ModelVersionCheckError(
                f"feature_version of {self.model_name} v{version}: {exc!r}"
            ) from exc

        if trained_on != FEATURE_VERSION:
            reason = (
                f"{self.model_name} v{version} was trained on feature version "
                f"{trained_on!r}; this API computes {FEATURE_VERSION!r}"
            )
            self._rejected[version] = reason
            log.error("incompatible_model_served", model_version=version, reason=reason)
            raise IncompatibleModelError(reason)
        self._verified.add(version)


def _parse(resp: httpx.Response) -> tuple[float, str | None]:
    """(fraud probability, model_version) from a V2 response for a batch of one."""
    try:
        body: Any = resp.json()
        data = body["outputs"][0]["data"]
        score = float(data[1])  # predict_proba row 0: [p(legit), p(fraud)]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        raise KServeResponseError(f"unexpected V2 response: {resp.text[:200]}") from exc
    if not 0.0 <= score <= 1.0:
        raise KServeResponseError(f"score out of range: {score}")
    version = body.get("model_version")
    return score, (str(version) if version is not None else None)
