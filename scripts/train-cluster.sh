#!/usr/bin/env bash
# Run a training Job in the cluster and wait for it (ADR-0007).
# One training path for both callers:
#   .github/workflows/train.yml   workflow_dispatch and weekly schedule
#   make train-cluster            by hand while the self-hosted runner is offline
#
# Uses the trainer image for this commit, building and pushing it if the
# registry does not have it yet. Registers a new fraud-detector version; it
# does not promote (run scripts/promote.py, which also rolls the predictor).
#
# Env: SEED (default: epoch seconds), CUSTOMERS (5000), DAYS (90),
#      REGISTRY (192.168.2.203:5000/fraud), NAMESPACE (fraud)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REGISTRY=${REGISTRY:-192.168.2.203:5000/fraud}
NAMESPACE=${NAMESPACE:-fraud}
SEED=${SEED:-$(date +%s)}
CUSTOMERS=${CUSTOMERS:-5000}
DAYS=${DAYS:-90}
SHA=$(git rev-parse --short HEAD)
IMAGE="$REGISTRY/trainer:$SHA"

registry_host=${REGISTRY%%/*}
repo_path=${REGISTRY#*/}/trainer
if ! curl -sf "http://$registry_host/v2/$repo_path/manifests/$SHA" \
     -H 'Accept: application/vnd.oci.image.index.v1+json, application/vnd.docker.distribution.manifest.v2+json' >/dev/null; then
  if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "train-cluster: $IMAGE is not in the registry and the tree is dirty; commit first" >&2
    exit 1
  fi
  echo "==> building $IMAGE"
  docker build -f ml/training/Dockerfile --build-arg GIT_SHA="$SHA" -t "$IMAGE" .
  docker push "$IMAGE"
fi

job=$(sed -e "s#TRAINER_IMAGE#$IMAGE#" -e "s#TRAIN_SEED#$SEED#" \
          -e "s#TRAIN_CUSTOMERS#$CUSTOMERS#" -e "s#TRAIN_DAYS#$DAYS#" deploy/jobs/train.yaml \
      | kubectl create -f - -o name)
echo "==> $job (image $IMAGE, seed $SEED, $CUSTOMERS customers, $DAYS days)"
[[ -n "${GITHUB_OUTPUT:-}" ]] && echo "job=${job#job.batch/}" >> "$GITHUB_OUTPUT"

# Follow the logs, then report the Job's own verdict.
until kubectl -n "$NAMESPACE" logs "$job" -f 2>/dev/null; do sleep 3; done
for _ in $(seq 1 60); do
  status=$(kubectl -n "$NAMESPACE" get "$job" -o jsonpath='{.status.succeeded}/{.status.failed}')
  case $status in
    1/*) echo "==> $job succeeded"; exit 0 ;;
    */1) echo "==> $job failed" >&2; exit 1 ;;
  esac
  sleep 5
done
echo "==> $job did not finish; check kubectl -n $NAMESPACE describe $job" >&2
exit 1
