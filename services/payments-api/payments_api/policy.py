"""Maps a fraud score to a decision."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Decision = Literal["approved", "review", "declined"]
DECISIONS: tuple[Decision, ...] = ("approved", "review", "declined")


@dataclass(frozen=True, slots=True)
class DecisionPolicy:
    review_threshold: float
    decline_threshold: float

    def decide(self, score: float) -> Decision:
        if score >= self.decline_threshold:
            return "declined"
        if score >= self.review_threshold:
            return "review"
        return "approved"
