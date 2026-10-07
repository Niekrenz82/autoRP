import mlflow
from mlflow import MlflowClient
from pathlib import Path

# Absolute paths, so every notebook/script uses the same store regardless of the working directory
# (Jupyter runs from notebooks/, where a relative "sqlite:///mlflow.db" would silently create a new database)
ROOT = Path(__file__).parents[1]
TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"  # runs, params, metrics and the model registry
ARTIFACT_URI = (ROOT / "mlruns").as_uri()                      # model files, tables and figures
EXPERIMENT = "pnc-pricing"


def setup():
    # Point MLflow at the project database and make the project experiment active
    mlflow.set_tracking_uri(TRACKING_URI)
    if mlflow.get_experiment_by_name(EXPERIMENT) is None:
        mlflow.create_experiment(EXPERIMENT, artifact_location=ARTIFACT_URI)
    mlflow.set_experiment(EXPERIMENT)


def promote(model_name, version, alias="baseline"):
    # Move an alias (e.g. "baseline") to a registered model version; the alias always points at exactly one version
    MlflowClient(TRACKING_URI).set_registered_model_alias(model_name, alias, str(version))


def load_model(model_name, alias="baseline"):
    # Load whichever registered version currently carries the alias
    setup()
    return mlflow.pyfunc.load_model(f"models:/{model_name}@{alias}")
