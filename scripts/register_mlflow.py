import csv
import os
import argparse
import mlflow

parser = argparse.ArgumentParser(description="Register MLflow run from CSV")

parser.add_argument("--experiment_id", default="3", help="MLflow experiment ID")
parser.add_argument("--name", required=True, help="Run name")
parser.add_argument("--csv", required=True, help="Path to CSV file with metrics")
parser.add_argument("--description", default="", help="Run description")
parser.add_argument("--model_type", default="", help="Tag: model_type")
parser.add_argument("--architecture", default="", help="Tag: architecture")
parser.add_argument("--input_type", default="", help="Tag: input_type")
parser.add_argument("--stage", default="", help="Tag: stage")
parser.add_argument("--model", default="", help="Param: model")
parser.add_argument("--dataset", default="", help="Param: dataset")
parser.add_argument("--optimizer", default="", help="Param: optimizer")

args = parser.parse_args()

with open("mlflow_credits.txt", "r") as f:
    os.environ["MLFLOW_TRACKING_USERNAME"] = f.readline().strip()
    os.environ["MLFLOW_TRACKING_PASSWORD"] = f.readline().strip()

mlflow.set_tracking_uri(
    "https://ai.swmansion.com/mlflow"
)

mlflow.set_experiment(
    experiment_id=args.experiment_id
)

with mlflow.start_run(
    run_name=args.name,
    description=args.description or None
):
    tags = {}
    if args.model_type:
        tags["model_type"] = args.model_type
    if args.architecture:
        tags["architecture"] = args.architecture
    if args.input_type:
        tags["input_type"] = args.input_type
    if args.stage:
        tags["stage"] = args.stage

    if tags:
        mlflow.set_tags(tags)

    params = {}
    if args.model:
        params["model"] = args.model
    if args.dataset:
        params["dataset"] = args.dataset
    if args.optimizer:
        params["optimizer"] = args.optimizer

    if params:
        mlflow.log_params(params)

    with open(args.csv, "r") as f:
        sep = csv.Sniffer().sniff(f.readline()).delimiter
        f.seek(0)
        reader = csv.DictReader(f, delimiter=sep)

        for row in reader:
            step = int(row["step"])

            if row["train_loss"]:
                mlflow.log_metric("train_loss", float(row["train_loss"]), step=step)

            if row["val_loss"]:
                mlflow.log_metric("val_loss", float(row["val_loss"]), step=step)

            if row["lr"]:
                mlflow.log_metric("learning_rate", float(row["lr"]), step=step)

    mlflow.log_artifact(args.csv)

print("Done")
