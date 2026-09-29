#!/bin/sh
# Create the artifact bucket if needed, then start MLflow with proxied artifacts.
set -eu

python - <<'PY'
import os

import boto3
from botocore.exceptions import ClientError

s3 = boto3.client("s3", endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"])
bucket = os.environ["MLFLOW_BUCKET"]
try:
    s3.head_bucket(Bucket=bucket)
except ClientError:
    s3.create_bucket(Bucket=bucket)
    print(f"created bucket {bucket}")
PY

exec mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --serve-artifacts \
    --artifacts-destination "s3://$MLFLOW_BUCKET" \
    --allowed-hosts "localhost:5000,127.0.0.1:5000,mlflow:5000"
