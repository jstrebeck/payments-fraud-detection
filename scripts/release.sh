#!/usr/bin/env bash
# Build and push this commit's images, then point deploy/overlays/homelab at them.
# One release path for both callers:
#   .github/workflows/build-push.yml   on every green CI run on main
#   make release                       by hand while the self-hosted runner is offline
# The caller commits and pushes the overlay change; Argo CD deploys it (ADR-0008).
#
# Usage: scripts/release.sh            (REGISTRY defaults to 192.168.2.203:5000/fraud)
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
REGISTRY=${REGISTRY:-192.168.2.203:5000/fraud}
OVERLAY=deploy/overlays/homelab/kustomization.yaml

if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "release: working tree has uncommitted changes; images are tagged by commit, so commit first" >&2
  exit 1
fi
SHA=$(git rev-parse --short HEAD)

declare -A DOCKERFILES=(
  [payments-api]=services/payments-api/Dockerfile
  [simulator]=services/simulator/Dockerfile
  [trainer]=ml/training/Dockerfile
)
for img in "${!DOCKERFILES[@]}"; do
  ref="$REGISTRY/$img:$SHA"
  echo "==> $ref"
  docker build -f "${DOCKERFILES[$img]}" --build-arg GIT_SHA="$SHA" -t "$ref" .
  docker push "$ref"
done

# The trainer image also runs the registry exporter, so it is pinned too;
# training Jobs pick their own tag per run (train.yml / make train-cluster).
python3 - "$OVERLAY" "$REGISTRY" "$SHA" <<'PY'
import re, sys
path, registry, sha = sys.argv[1:]
text = open(path).read()
for img in ("payments-api", "simulator", "trainer"):
    pattern = rf"(- name: {re.escape(registry)}/{img}\n\s+newTag: )\S+"
    text, n = re.subn(pattern, rf"\g<1>{sha}", text)
    if n != 1:
        sys.exit(f"release: no images entry for {registry}/{img} in {path}")
open(path, "w").write(text)
PY
echo "==> $OVERLAY now pins $SHA"
