# Local MLflow server for docker-compose only. The cluster MLflow lives in the
# homelab repo. The upstream image lacks the Postgres and S3 client libraries.
FROM ghcr.io/mlflow/mlflow:v3.16.1
RUN pip install --no-cache-dir psycopg2-binary==2.9.13 boto3==1.43.104
COPY start.sh /usr/local/bin/start-mlflow
ENTRYPOINT ["/usr/local/bin/start-mlflow"]
