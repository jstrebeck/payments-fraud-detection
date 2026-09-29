"""MlflowScorer: the registry's aliased model, loaded in-process (Phase 2).

Used locally and in compose; the cluster uses KServe from Phase 4. The model
takes the `ml.features` vector as named columns and returns predict_proba
(ADR-0013). A model whose `feature_version` tag differs from this build's
`FEATURE_VERSION` is refused, because it would be scored on features it was
not trained on.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import structlog
from mlflow import MlflowClient

from ml.features import FEATURE_NAMES, FEATURE_VERSION
from payments_api import metrics
from payments_api.scoring.base import ScoreResult

log = structlog.get_logger(__name__)


class ModelUnavailableError(RuntimeError):
    """No compatible model is loaded; the caller falls back to rules."""


class IncompatibleModelError(RuntimeError):
    """The aliased model was trained on a different feature version."""


@dataclass(frozen=True, slots=True)
class LoadedModel:
    version: str
    model: Any  # mlflow.pyfunc.PyFuncModel


class MlflowScorer:
    name = "mlflow"

    def __init__(
        self,
        *,
        tracking_uri: str | None,
        model_name: str,
        alias: str,
        refresh_seconds: float,
    ) -> None:
        self.model_name = model_name
        self.alias = alias
        self.refresh_seconds = refresh_seconds
        self._tracking_uri = tracking_uri
        self._loaded: LoadedModel | None = None
        self._task: asyncio.Task[None] | None = None

    @property
    def model_version(self) -> str:
        return f"{self.model_name}/{self._loaded.version}" if self._loaded else "unloaded"

    async def start(self) -> None:
        await self.refresh()
        if self.refresh_seconds > 0:
            self._task = asyncio.create_task(self._refresh_loop())

    async def close(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task

    async def ready(self) -> bool:
        return self._loaded is not None

    async def refresh(self) -> bool:
        """Load the aliased version if it changed. Returns True when a new model went live.

        Failures keep the current model (or none) and are logged; they never raise.
        """
        try:
            loaded = await asyncio.to_thread(self._load_if_changed)
        except Exception as exc:
            log.warning(
                "model_load_failed",
                model=self.model_name,
                alias=self.alias,
                error=repr(exc),
                current=self.model_version,
            )
            return False
        if loaded is None:
            return False
        previous = self.model_version
        self._loaded = loaded
        metrics.set_model_version(self.name, self.model_version)
        log.info("model_loaded", model_version=self.model_version, previous=previous)
        return True

    async def score(self, features: Mapping[str, float]) -> ScoreResult:
        loaded = self._loaded  # one snapshot per request, even if a swap happens meanwhile
        if loaded is None:
            raise ModelUnavailableError(f"no {self.model_name}@{self.alias} loaded")
        frame = pd.DataFrame([[features[n] for n in FEATURE_NAMES]], columns=list(FEATURE_NAMES))
        out = np.asarray(await asyncio.to_thread(loaded.model.predict, frame))
        return ScoreResult(
            score=float(out[0, 1]),
            scorer=self.name,
            model_version=f"{self.model_name}/{loaded.version}",
        )

    async def _refresh_loop(self) -> None:
        while True:
            await asyncio.sleep(self.refresh_seconds)
            await self.refresh()

    def _load_if_changed(self) -> LoadedModel | None:
        client = MlflowClient(self._tracking_uri)
        mv = client.get_model_version_by_alias(self.model_name, self.alias)
        version = str(mv.version)  # int from SQL-backed stores, str over REST
        if self._loaded is not None and version == self._loaded.version:
            return None
        trained_on = mv.tags.get("feature_version")
        if trained_on != FEATURE_VERSION:
            raise IncompatibleModelError(
                f"{self.model_name} v{version} was trained on feature version "
                f"{trained_on!r}; this API computes {FEATURE_VERSION!r}"
            )
        if self._tracking_uri:
            mlflow.set_tracking_uri(self._tracking_uri)
        model = mlflow.pyfunc.load_model(f"models:/{self.model_name}/{version}")
        return LoadedModel(version=version, model=model)
