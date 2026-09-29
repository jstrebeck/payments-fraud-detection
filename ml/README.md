# ml

The machine learning package: data generation, features, training,
evaluation, and (if ever needed) custom serving code. A single installable
package `ml` with optional dependency groups so the API can import
`ml.features` without pulling in numpy, LightGBM or MLServer.

```
ml/
  data/        synthetic generator and the canonical transaction schema   (Phase 1)
  features/    pure feature functions + batch featuriser                  (Phase 1)
  training/    train entrypoint (Job image), MLflow logging, registration  (Phase 2)
  evaluation/  metrics, champion/challenger gate, model card; drift later  (Phase 2)
  serving/     KServe transformer or custom predictor; empty unless needed (Phase 4)
  tests/
  pyproject.toml
  Dockerfile   trainer image (Phase 5)
```

This directory *is* the `ml` package (there is no `src/`): `pyproject.toml`
maps the package name onto `.` with setuptools' `package-dir`. Add each new
subpackage to `packages` there.

## Dependency groups

| Group | Used by | Contents |
|---|---|---|
| core | everyone, including the API image | pydantic |
| `data` | generator, batch featuriser, simulator, trainer | numpy, pyarrow |
| `score` | API (`MlflowScorer`) | mlflow-skinny 3.x (matches the homelab server), lightgbm, scikit-learn, skops, pandas |
| `train` | trainer, `scripts/promote.py` | `data` + `score` + pydantic-settings, pyyaml |
| `eval` (Phase 2) | trainer, drift job | evidently (or none), matplotlib |
| `serve` (Phase 4, if needed) | ml/serving only | mlserver, mlserver-mlflow |

Modules that need an extra say so in their docstring. `ml.data.schema`,
`ml.data.reference` and all of `ml.features` except `ml.features.batch`
import with core only.

## Invariants

- `ml.data.schema` is the only definition of a transaction.
- `ml.features` has no I/O and no dependency on the API or the database.
- Given a seed, `ml.data` produces byte-identical Parquet across runs; a test
  asserts it, and a golden summary test catches unintended generator changes.

## Tests

`uv run pytest ml` (or `make test` for everything).
