"""Feature drift: a reference profile of the training data and PSI against live traffic.

The trainer logs `build_profile(train_features)` as the run artifact
`reference/feature_profile.json`. The drift monitor (`ml.evaluation.drift_monitor`)
bins recent live feature vectors the same way and reports the Population
Stability Index per feature:

    PSI = sum over bins of (live% - ref%) * ln(live% / ref%)

with both proportions floored at `EPSILON` so an empty bin does not blow up.
Rule of thumb: < 0.1 stable, 0.1 to 0.25 moderate shift, > 0.25 major shift.

Binning, per feature:

- **categorical** (at most `MAX_CATEGORIES` distinct training values: the 0/1
  novelty flags, category, risk-tier and channel codes, small counts): one bin
  per training value plus an `other` bin for values never seen in training.
- **continuous**: decile edges of the training values, deduplicated (a feature
  that is mostly 0 gets fewer bins), with open-ended first and last bins, so
  live values outside the training range still land somewhere.

Pure functions only; everything is deterministic for a given frame.

    python -m ml.evaluation.drift profile --version 2

rebuilds the profile for an already registered version (regenerating its
training data from the run's logged generator params) and logs it to that
version's run. Used to backfill champions trained before profiles existed.
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

PROFILE_ARTIFACT = "reference/feature_profile.json"
PROFILE_FORMAT = 1
EPSILON = 1e-4
MAX_CATEGORIES = 32
DECILES = tuple(q / 10 for q in range(1, 10))
DECIMALS = 6  # categorical values are matched after rounding


@dataclass(frozen=True, slots=True)
class FeatureProfile:
    kind: Literal["categorical", "continuous"]
    # categorical: the training values; continuous: the inner bin edges
    points: list[float]
    # Expected share per bin. categorical: one per value, then `other`;
    # continuous: len(points) + 1 bins, (-inf, e0], (e0, e1], ..., (e_last, inf)
    expected: list[float]


@dataclass(frozen=True, slots=True)
class Profile:
    n: int
    features: dict[str, FeatureProfile]
    format: int = PROFILE_FORMAT

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Profile:
        if data.get("format") != PROFILE_FORMAT:
            raise ValueError(f"unsupported profile format {data.get('format')!r}")
        return cls(
            n=int(data["n"]),
            features={name: FeatureProfile(**fp) for name, fp in data["features"].items()},
        )


def _finite(values: pd.Series) -> np.ndarray:
    arr = values.to_numpy(dtype="float64")
    return arr[np.isfinite(arr)]


def _bin_counts(fp: FeatureProfile, values: np.ndarray) -> np.ndarray:
    if fp.kind == "categorical":
        index = {round(v, DECIMALS): i for i, v in enumerate(fp.points)}
        other = len(fp.points)
        idx = np.fromiter(
            (index.get(round(float(v), DECIMALS), other) for v in values), dtype=np.int64
        )
        return np.bincount(idx, minlength=len(fp.points) + 1)
    idx = np.searchsorted(np.asarray(fp.points, dtype="float64"), values, side="left")
    return np.bincount(idx, minlength=len(fp.points) + 1)


def _shares(counts: np.ndarray) -> list[float]:
    total = counts.sum()
    if total == 0:
        return [0.0] * len(counts)
    return [float(c) / float(total) for c in counts]


def build_feature(values: pd.Series) -> FeatureProfile:
    arr = _finite(values)
    distinct = np.unique(np.round(arr, DECIMALS))
    if len(distinct) <= MAX_CATEGORIES:
        fp = FeatureProfile("categorical", [float(v) for v in distinct], [])
    else:
        edges = np.unique(np.quantile(arr, DECILES))
        fp = FeatureProfile("continuous", [float(e) for e in edges], [])
    return FeatureProfile(fp.kind, fp.points, _shares(_bin_counts(fp, arr)))


def build_profile(df: pd.DataFrame) -> Profile:
    """Reference profile of every column of a feature frame (the training split)."""
    return Profile(n=len(df), features={col: build_feature(df[col]) for col in df.columns})


def psi_feature(fp: FeatureProfile, values: pd.Series) -> float:
    actual = _shares(_bin_counts(fp, _finite(values)))
    total = 0.0
    for exp, act in zip(fp.expected, actual, strict=True):
        e, a = max(exp, EPSILON), max(act, EPSILON)
        total += (a - e) * math.log(a / e)
    return total


def psi(profile: Profile, live: pd.DataFrame) -> dict[str, float]:
    """PSI per profiled feature present in `live`. Features missing from `live` are skipped."""
    return {
        name: psi_feature(fp, live[name])
        for name, fp in profile.features.items()
        if name in live.columns
    }


# --- MLflow artifacts over the proxy ---------------------------------------------


def fetch_run_artifact(tracking_uri: str, artifact_uri: str, path: str, timeout: float = 30) -> Any:
    """Read a JSON artifact of a run through MLflow's artifact proxy (`--serve-artifacts`).

    Plain HTTP on purpose: the MLflow client's parallel downloader stalled against
    the homelab server and left zero-filled files (ADR-0006).
    """
    import re
    import urllib.parse
    import urllib.request

    match = re.match(r"^mlflow-artifacts:(?://[^/]*)?/?(?P<root>.*)$", artifact_uri)
    if not match:
        raise ValueError(f"{artifact_uri!r} is not served by the MLflow artifact proxy")
    root = match["root"].strip("/")
    url = (
        f"{tracking_uri.rstrip('/')}/api/2.0/mlflow-artifacts/artifacts/"
        f"{urllib.parse.quote(f'{root}/{path}')}"
    )
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.load(resp)


# --- backfill CLI ---------------------------------------------------------------


def _generator_config_from_params(params: Mapping[str, str]) -> Any:
    """GeneratorConfig for a training run, from its logged `data.*` params."""
    from datetime import datetime

    from ml.data.generator import GeneratorConfig

    data = {k.removeprefix("data."): v for k, v in params.items() if k.startswith("data.")}
    kwargs: dict[str, Any] = {
        "seed": int(data["seed"]),
        "customers": int(data["customers"]),
        "days": int(data["days"]),
        "fraud_rate": float(data["fraud_rate"]),
        "start": datetime.fromisoformat(data["start"]),
    }
    if data.get("merchants") not in (None, "None"):
        kwargs["merchants"] = int(data["merchants"])
    return GeneratorConfig(**kwargs)


def backfill(model: str, version: str, *, dry_run: bool = False) -> dict[str, Any]:
    """Rebuild the reference profile of a registered version and log it to its run."""
    import mlflow
    from mlflow import MlflowClient

    from ml.data.generator import GENERATOR_VERSION, generate
    from ml.data.io import write_parquet
    from ml.training.config import SplitConfig
    from ml.training.data import load_dataset, time_split

    client = MlflowClient()
    mv = client.get_model_version(model, version)
    if not mv.run_id:
        raise RuntimeError(f"{model} v{version} has no training run")
    run_id: str = mv.run_id
    run = client.get_run(run_id)
    params = run.data.params
    logged_generator = params.get("data.generator_version")
    if logged_generator is not None and logged_generator != GENERATOR_VERSION:
        raise RuntimeError(
            f"v{version} was trained on generator version {logged_generator}; this code "
            f"generates version {GENERATOR_VERSION}, so its training data cannot be rebuilt"
        )
    config = _generator_config_from_params(params)
    # The split the run used is in its logged train_config.json.
    tracking_uri = mlflow.get_tracking_uri()
    train_config = fetch_run_artifact(tracking_uri, run.info.artifact_uri, "train_config.json")
    split_cfg = SplitConfig(**train_config["split"])
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "transactions.parquet"
        write_parquet(generate(config), path)
        dataset, _ = load_dataset(path)
    train = time_split(dataset, split_cfg).train
    if params.get("n_train") and int(params["n_train"]) != len(train):
        raise RuntimeError(
            f"regenerated training split has {len(train)} rows, the run logged "
            f"{params['n_train']}: generator or split changed since v{version} was trained"
        )
    profile = build_profile(train.features)
    if not dry_run:
        client.log_dict(run_id, profile.to_dict(), PROFILE_ARTIFACT)
    return {"version": version, "run_id": run_id, "n": profile.n, "logged": not dry_run}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Feature drift reference profiles")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("profile", help="rebuild and log the reference profile of a version")
    p.add_argument("--model", default="fraud-detector")
    p.add_argument("--version", required=True)
    p.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    print(json.dumps(backfill(args.model, args.version, dry_run=args.dry_run)))


if __name__ == "__main__":
    main()
