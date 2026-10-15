"""Log the exact DVC input data fingerprint and pipeline state to MLflow."""

import json
import subprocess

import mlflow
import pandas as pd
import yaml


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp3-dvc")

with open("params.yaml", encoding="utf-8") as stream:
    params = yaml.safe_load(stream)
with open("data/transactions.csv.dvc", encoding="utf-8") as stream:
    ticket = yaml.safe_load(stream)["outs"][0]
with open("reports/metrics.json", encoding="utf-8") as stream:
    metrics = json.load(stream)

commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"], text=True
).strip()
row_count = len(pd.read_csv("data/transactions.csv"))

with mlflow.start_run(run_name=f"donnees-{ticket['md5'][:8]}"):
    mlflow.log_params(params["entrainement"])
    mlflow.log_metrics(metrics)
    mlflow.set_tags(
        {
            "commit": commit,
            "donnees_md5": ticket["md5"],
            "donnees_lignes": row_count,
        }
    )
    mlflow.log_artifact("dvc.lock")

print(f"run MLflow cree : donnees {ticket['md5'][:8]} - {row_count} lignes")
