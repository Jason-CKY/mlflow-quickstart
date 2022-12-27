import os
import numpy as np
import tempfile
from pprint import pprint

import mlflow
from mlflow.tracking import MlflowClient
import torch
import requests
import random

metrics = {
    "test-int": 2,
    "test-float": 4.23,
    'test-np': np.float64(2.41245345)
}

metrics2 = {
    "test-int": 2,
    "test-float": 4.23
}


params = {
    "test-param": random.random() * random.randint(1, 500),
    "test-param1": random.random() * random.randint(1, 500),
    "test-param2": random.random() * random.randint(1, 500),
    "test-param3": random.random() * random.randint(1, 500),
    "test-param4": random.random() * random.randint(1, 500)
}

artifact_path = "/app/imagenet_classes.txt"

def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)

def upload_model():
    assert "MLFLOW_TRACKING_URI" in os.environ
    model_name = 'imagenet'
    client = MlflowClient()
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
    with mlflow.start_run() as run:
        mlflow.log_artifact(artifact_path)
        mlflow.log_metrics(metrics)
        mlflow.pytorch.log_model(model, "resnet18-imagenet", registered_model_name=model_name)

    with mlflow.start_run() as run:
        mlflow.log_artifact(artifact_path)
        mlflow.log_metrics(metrics2)
        mlflow.pytorch.log_model(model, "resnet18-imagenet", registered_model_name=model_name)

    with mlflow.start_run() as run:
        mlflow.log_artifact(artifact_path)
        mlflow.log_metrics(metrics)
        mlflow.log_params(params)
        mlflow.pytorch.log_model(model, "resnet18-imagenet", registered_model_name=model_name)

    client.transition_model_version_stage(model_name, "1", "production")
    client.transition_model_version_stage(model_name, "2", "staging")

    client.delete_run(run.info.run_id)

#  NOTE: ensure the tracking server has been started with --serve-artifacts to enable
#        MLflow artifact serving functionality.


def main():
    assert "MLFLOW_TRACKING_URI" in os.environ

    # Upload artifacts
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path_a = os.path.join(tmp_dir, "a.txt")
        save_text(tmp_path_a, "0")
        tmp_sub_dir = os.path.join(tmp_dir, "dir")
        os.makedirs(tmp_sub_dir)
        tmp_path_b = os.path.join(tmp_sub_dir, "b.txt")
        save_text(tmp_path_b, "1")
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifacts(tmp_sub_dir, artifact_path="dir")

    # Download artifacts
    client = mlflow.tracking.MlflowClient()
    pprint(os.listdir(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="")))
    pprint(os.listdir(mlflow.artifacts.download_artifacts(run_id=run.info.run_id, artifact_path="dir")))

    # List artifacts
    pprint(client.list_artifacts(run.info.run_id))
    pprint(client.list_artifacts(run.info.run_id, "dir"))


if __name__ == "__main__":
    main()
    upload_model()