# Mlflow quick-setup

Quickly setup mlflow with remote tracking server and s3 artifact store. Just run `docker-compose up` to get started.

```bash
source .env
MLFLOW_TRACKING_URI=http://localhost:5000 mlflow runs list --experiment-id 0 --view deleted_only
MLFLOW_TRACKING_URI=http://localhost:5000 mlflow gc --backend-store-uri postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:5432/${POSTGRES_DB}  --run-ids 

```
