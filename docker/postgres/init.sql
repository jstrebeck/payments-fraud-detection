-- Runs once, on first start of an empty data volume. The `payments` database
-- is created by POSTGRES_DB; MLflow gets its own database.
CREATE DATABASE mlflow OWNER fraud;
