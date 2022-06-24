import os
import tempfile
from pprint import pprint

import mlflow

def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)

def main():
    assert "MLFLOW_TRACKING_URI" in os.environ
    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path_a = os.path.join(tmp_dir, "a.txt")
        save_text(tmp_path_a, "0")
        tmp_sub_dir = os.path.join(tmp_dir, "dir")
        os.makedirs(tmp_sub_dir)
        tmp_path_b = os.path.join(tmp_sub_dir, "b.txt")
        save_text(tmp_path_b, "1")
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifacts(tmp_sub_dir, artifact_path="dir")
        mlflow.log_metrics({"test": 2})

    client = mlflow.tracking.MlflowClient()
    client.delete_run(run.info.run_id)

    with mlflow.start_run() as run, tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path_a = os.path.join(tmp_dir, "a.txt")
        save_text(tmp_path_a, "0")
        tmp_sub_dir = os.path.join(tmp_dir, "dir")
        os.makedirs(tmp_sub_dir)
        tmp_path_b = os.path.join(tmp_sub_dir, "b.txt")
        save_text(tmp_path_b, "1")
        mlflow.log_artifact(tmp_path_a)
        mlflow.log_artifacts(tmp_sub_dir, artifact_path="dir")
        mlflow.log_metrics({"test": 2})

    # Downloading artifacts
    pprint(os.listdir(client.download_artifacts(run.info.run_id, "")))
    pprint(os.listdir(client.download_artifacts(run.info.run_id, "dir")))

    # List artifacts
    pprint(client.list_artifacts(run.info.run_id))
    pprint(client.list_artifacts(run.info.run_id, "dir"))

if __name__ == "__main__":
    main()
