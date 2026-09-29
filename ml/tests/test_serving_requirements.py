"""The serving image must pin exactly the library versions that train the model.

A model logged by one scikit-learn/skops/LightGBM version and loaded by another
can fail to load or score differently. ml/serving/requirements.in is the
serving image's source of truth; this test ties it to the training
environment (the uv lock this test runs in).
"""

from __future__ import annotations

import re
from importlib.metadata import version
from pathlib import Path

import pytest

from ml.training.train import MODEL_REQUIREMENTS

REQUIREMENTS_IN = Path(__file__).resolve().parents[1] / "serving" / "requirements.in"
PIN = re.compile(r"^([A-Za-z0-9_.-]+)==([^\s#]+)")


def _serving_pins() -> dict[str, str]:
    pins: dict[str, str] = {}
    for line in REQUIREMENTS_IN.read_text().splitlines():
        match = PIN.match(line.strip())
        if match:
            pins[match.group(1).lower()] = match.group(2)
    return pins


@pytest.mark.parametrize("package", [*MODEL_REQUIREMENTS, "mlflow-skinny"])
def test_serving_pin_matches_training_env(package: str) -> None:
    pins = _serving_pins()
    assert package in pins, f"{package} is not pinned in {REQUIREMENTS_IN}"
    assert pins[package] == version(package), (
        f"{package}: serving image pins {pins[package]}, training uses {version(package)}. "
        "Update ml/serving/requirements.in, recompile requirements.txt and rebuild the image."
    )
