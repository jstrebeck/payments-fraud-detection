"""Run the promotion gate for a registered model version (Phase 2).

Evaluates the version and the current `champion` on the version's own test
window, and moves `champion` (keeping the old one as `previous`) only if the
gate rule in ml/training/config.yaml passes. The only supported way to move
`champion` (ADR-0004, ADR-0012).

After a promotion it rolls the predictor (ADR-0006): the InferenceService's
storageUri is `models:/<model>@champion`, resolved when a pod starts, so new
pods are needed. It sets the pod annotation `fraud-detection/model-version` on
the InferenceService, which KServe turns into a rolling update. (A plain
`kubectl rollout restart` does not work: KServe reverts it.) Needs `kubectl`
with access to the namespace; `--no-rollout` skips it.

    uv run python scripts/promote.py                # latest version
    uv run python scripts/promote.py --version 7 --dry-run

Reads MLFLOW_TRACKING_URI and MODEL_NAME (default fraud-detector) from the
environment. Exit codes: 0 promoted or rejected, 3 rejected with --strict,
4 promoted but the predictor rollout failed (the alias has moved; rerun with
--rollout-only once kubectl works).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow import MlflowClient

from ml.evaluation.promote import run_gate
from ml.training.config import TrainConfig

ROLLOUT_ANNOTATION = "fraud-detection/model-version"


def rollout(namespace: str, isvc: str, version: str) -> bool:
    """Roll the predictor so its pods resolve the alias again. True on success."""
    patch = {"spec": {"predictor": {"annotations": {ROLLOUT_ANNOTATION: version}}}}
    steps = [
        ["kubectl", "-n", namespace, "patch", "inferenceservice", isvc,
         "--type", "merge", "-p", json.dumps(patch)],
        ["kubectl", "-n", namespace, "rollout", "status",
         f"deployment/{isvc}-predictor", "--timeout=5m"],
        ["kubectl", "-n", namespace, "wait", f"inferenceservice/{isvc}",
         "--for=condition=Ready", "--timeout=2m"],
    ]  # fmt: skip
    for cmd in steps:
        try:
            subprocess.run(cmd, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"rollout    FAILED: {' '.join(cmd[:5])}: {exc}", file=sys.stderr)
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "fraud-detector"))
    parser.add_argument("--version", help="registered version to gate (default: latest)")
    parser.add_argument("--config", type=Path, help="training config with the gate rule")
    parser.add_argument("--dry-run", action="store_true", help="evaluate, do not move aliases")
    parser.add_argument("--strict", action="store_true", help="exit 3 when the gate rejects")
    parser.add_argument("--no-rollout", action="store_true", help="move the alias only")
    parser.add_argument(
        "--rollout-only",
        action="store_true",
        help="skip the gate; roll the predictor to the current champion",
    )
    parser.add_argument("--namespace", default=os.environ.get("KSERVE_NAMESPACE", "fraud"))
    parser.add_argument("--isvc", default=os.environ.get("KSERVE_ISVC", "fraud-detector"))
    args = parser.parse_args()

    client = MlflowClient()
    if args.rollout_only:
        champion = str(client.get_model_version_by_alias(args.model, "champion").version)
        print(f"rollout    {args.namespace}/{args.isvc} -> {args.model} v{champion}")
        return 0 if rollout(args.namespace, args.isvc, champion) else 4

    version = args.version or str(
        max(int(v.version) for v in client.search_model_versions(f"name='{args.model}'"))
    )
    outcome = run_gate(
        client,
        args.model,
        version,
        TrainConfig.load(args.config).gate.rule(),
        apply=not args.dry_run,
    )
    r = outcome.result
    print(f"model      {args.model} v{version}")
    print(f"gate       {'PASS' if r.promote else 'FAIL'}: {r.reason}")
    print(f"champion   {outcome.champion_before or 'none'} -> {outcome.champion_after or 'none'}"
          + ("  (dry run)" if args.dry_run else ""))  # fmt: skip
    moved = not args.dry_run and outcome.champion_after != outcome.champion_before
    if moved and not args.no_rollout and outcome.champion_after:
        print(f"rollout    {args.namespace}/{args.isvc} -> {args.model} v{outcome.champion_after}")
        if not rollout(args.namespace, args.isvc, str(outcome.champion_after)):
            return 4
    return 3 if args.strict and not r.promote else 0


if __name__ == "__main__":
    sys.exit(main())
