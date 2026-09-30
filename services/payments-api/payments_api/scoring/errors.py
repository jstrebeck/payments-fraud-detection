"""Scorer errors. Their class names become `fraud_scorer_fallback_total{reason}` labels."""

from __future__ import annotations


class ModelUnavailableError(RuntimeError):
    """No compatible model is loaded or reachable; the caller falls back to rules."""


class IncompatibleModelError(RuntimeError):
    """The model was trained on a different feature version, or its version is unknown."""
