"""The FraudScorer interface (ADR-0001)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class ScoreResult:
    score: float  # probability-like, 0..1
    scorer: str
    model_version: str
    # Decision thresholds that belong to the model that produced the score (its
    # model card's recommendation). None: use the API's configured policy.
    review_threshold: float | None = None
    decline_threshold: float | None = None


class FraudScorer(Protocol):
    name: str

    @property
    def model_version(self) -> str:
        """What produced the scores right now (a rules version or name/version)."""
        ...

    async def score(self, features: Mapping[str, float]) -> ScoreResult:
        """Score one feature vector. May raise; the caller falls back."""
        ...

    async def ready(self) -> bool:
        """Whether the scorer can currently serve requests."""
        ...

    async def start(self) -> None:
        """Called once at app startup (load models, start background refresh)."""
        ...

    async def close(self) -> None:
        """Called once at app shutdown."""
        ...
