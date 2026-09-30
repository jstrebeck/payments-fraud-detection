# Architecture Decision Records

One file per decision, numbered, never deleted. To reverse a decision, write a
new ADR that supersedes the old one and link both ways. Copy
`0000-template.md` to start.

| ADR | Decision | Status |
|---|---|---|
| [0001](0001-payments-api-as-model-consumer.md) | FastAPI payments API is the model consumer, not the subject | Accepted |
| [0002](0002-two-repo-boundary.md) | Platform components live in the homelab repo; workloads live here | Accepted |
| [0003](0003-synthetic-data.md) | Synthetic, seeded data instead of a public dataset | Accepted |
| [0004](0004-mlflow-tracking-and-registry.md) | MLflow for tracking and registry, alias-based promotion | Accepted (model format superseded by 0013) |
| [0005](0005-kserve-rawdeployment.md) | KServe in RawDeployment mode, MLflow runtime, V2 protocol | Accepted (runtime image superseded by 0014) |
| [0006](0006-model-promotion-mechanism.md) | How a promoted model reaches the InferenceService | Accepted: `models:/` alias resolved at pod start, `promote.py` rolls the predictor |
| [0007](0007-orchestration-kubernetes-jobs.md) | Kubernetes Jobs/CronJobs before any workflow engine | Accepted |
| [0008](0008-gitops-argocd.md) | Argo CD for delivery | Accepted |
| [0009](0009-python-tooling.md) | uv workspace, ruff, mypy, pytest, Python 3.12 | Accepted |
| [0010](0010-card-context-window.md) | One card-history window definition shared by training and serving | Accepted |
| [0011](0011-seaweedfs-s3.md) | SeaweedFS instead of MinIO for S3-compatible storage | Accepted |
| [0012](0012-shared-homelab-mlflow.md) | Development uses the homelab MLflow; local MLflow behind a compose profile | Accepted |
| [0013](0013-model-input-is-the-feature-vector.md) | The model's input is the feature vector; `feature_version` handshake | Accepted (serving runtime superseded by 0014) |
| [0014](0014-custom-serving-runtime.md) | Custom MLServer image as the KServe serving runtime | Accepted |
