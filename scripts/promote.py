"""Run the promotion gate for a registered model version (Phase 2).

Evaluates the version and the current `champion` on the version's own test
window, and moves `champion` (keeping the old one as `previous`) only if the
gate rule in ml/training/config.yaml passes. The only supported way to move
`champion` (ADR-0004, ADR-0012).

    uv run python scripts/promote.py                # latest version
    uv run python scripts/promote.py --version 7 --dry-run

Reads MLFLOW_TRACKING_URI and MODEL_NAME (default fraud-detector) from the
environment. Exit codes: 0 promoted or rejected, 3 rejected with --strict.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from mlflow import MlflowClient

from ml.evaluation.promote import run_gate
from ml.training.config import TrainConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default=os.environ.get("MODEL_NAME", "fraud-detector"))
    parser.add_argument("--version", help="registered version to gate (default: latest)")
    parser.add_argument("--config", type=Path, help="training config with the gate rule")
    parser.add_argument("--dry-run", action="store_true", help="evaluate, do not move aliases")
    parser.add_argument("--strict", action="store_true", help="exit 3 when the gate rejects")
    args = parser.parse_args()

    client = MlflowClient()
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
    return 3 if args.strict and not r.promote else 0


if __name__ == "__main__":
    sys.exit(main())
