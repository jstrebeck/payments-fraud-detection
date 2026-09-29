# ml.training

Entry point of the trainer (`python -m ml.training`, `make train`; a
Kubernetes Job image from Phase 5). One process, staged, with MLflow as the
record (ADR-0007). Needs the `train` extra.

## Stages

1. **Load.** Parquet from `DATA_URI`; every row is validated against
   `ml.data.schema` (Phase 7 adds labelled decisions from Postgres).
2. **Featurise.** `ml.features.batch.featurize`, the same code the API runs
   (ADR-0010). Done on the whole stream before splitting, so rows near a
   boundary still see their card's earlier history, as they would online.
3. **Split.** By time: train (oldest 70%), validation (next 15%, early
   stopping), test (most recent 15%, used for the report and the gate).
4. **Fit.** `LGBMClassifier` on the feature vector (ADR-0013), no class
   reweighting so scores stay roughly calibrated probabilities. Parameters
   from `config.yaml`, logged to MLflow.
5. **Evaluate.** `ml.evaluation.evaluate` on the test window; metrics,
   `evaluation/report.json`, feature importance and the test set are logged.
6. **Log and register.** `mlflow.sklearn.log_model` (skops format, signature,
   input example, `predict_proba`, pinned `pip_requirements`), registered as
   `MODEL_NAME`. Version tags: `feature_version`, `generator_version`, `seed`,
   `git_sha`, `data_uri`. The rendered model card is `model_card.md`.
7. **Gate** (only with `PROMOTE=true`): `ml.evaluation.promote.run_gate`, the
   same code as `make promote`.

Deterministic: same data and config give the same model and metrics.

## Configuration

Environment (`pydantic-settings`, also read from `.env`):

| Variable | Default | |
|---|---|---|
| `MLFLOW_TRACKING_URI` | homelab MLflow via the Makefile | read by MLflow |
| `DATA_URI` | `data/transactions.parquet` | |
| `MLFLOW_EXPERIMENT_NAME` | `fraud-detector-dev` | ADR-0012 |
| `MODEL_NAME` | `fraud-detector` | |
| `TRAIN_CONFIG` | packaged `config.yaml` | model params, split, gate rule |
| `PROMOTE` / `PROMOTE_STRICT` | `false` / `false` | run the gate after training / exit 3 on rejection |
| `GIT_SHA` | `git rev-parse` | set in images |

## Exit codes

`0` success, `2` data missing or invalid, `3` gate rejected with
`PROMOTE_STRICT=true` (a rejection is otherwise a normal outcome).

## Current numbers

Generator v2, seed 42, 5000 customers, 90 days: PR-AUC 0.924, recall 0.951 at
1% FPR; per pattern 0.88 (card testing) to 0.99 (high value). About 40 s on a
laptop.
