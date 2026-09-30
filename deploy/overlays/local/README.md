# overlays/local

Not implemented. Without the homelab, the local path is `docker compose`
(`make dev`, `make simulate`), which runs the same images with the in-process
`MlflowScorer`. A kind/minikube overlay would also need KServe, cert-manager,
MLflow and Postgres installed, which is out of scope for now.
