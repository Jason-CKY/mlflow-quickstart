import os
import numpy as np

import mlflow
from mlflow.tracking import MlflowClient

def upload_model():
    assert "MLFLOW_TRACKING_URI" in os.environ
    client = MlflowClient()
    with mlflow.start_run() as run:
        mlflow.log_metric("delete-me", np.float64(0.415616846531434863123678))

    client.delete_run(run.info.run_id)

if __name__ == "__main__":
    upload_model()

