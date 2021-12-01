import os
import pickle
from pathlib import Path

import mlflow
import pytest

from omigami.spectra_matching.predictor import Predictor
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway


@pytest.fixture(scope="module")
def mlflow_setup(tmpdir_factory):
    tmpdir = tmpdir_factory.mktemp("model")
    mlflow_uri = f"sqlite:///{tmpdir}/mlflow.sqlite"
    dgw = MLFlowDataGateway(mlflow_uri)
    dgw._get_or_create_experiment_id("test-experiment", str(tmpdir))
    model = Predictor()
    return {"model": model, "path": tmpdir, "dgw": dgw, "experiment": "test-experiment"}


def test_get_or_create_experiment(mlflow_setup):
    dgw: MLFlowDataGateway = mlflow_setup["dgw"]
    path = str(mlflow_setup["path"])
    experiment_name = "new-experiment"

    # create
    created_experiment_id = dgw._get_or_create_experiment_id(experiment_name, path)

    # get
    experiment_id = dgw._get_or_create_experiment_id(experiment_name, path)
    assert created_experiment_id == experiment_id


def test_register_model(mlflow_setup):
    dgw: MLFlowDataGateway = mlflow_setup["dgw"]
    path = mlflow_setup["path"]

    run_id = dgw.register_model(
        model=mlflow_setup["model"],
        run_name="run",
        experiment_name=mlflow_setup["experiment"],
        model_name="test",
    )

    assert mlflow_setup["model"]._run_id == run_id
    model_files = os.listdir(f"{path}/{run_id}/artifacts/model")
    assert set(model_files) == {"MLmodel", "code", "python_model.pkl", "conda.yaml"}
    assert "spectra_matching" in os.listdir(
        f"{path}/{run_id}/artifacts/model/code/omigami/"
    )


def test_register_model_with_params(mlflow_setup):
    dgw: MLFlowDataGateway = mlflow_setup["dgw"]

    super_params = {"param1": "fire", "param2": "water", "param3": "weed"}
    run_id = dgw.register_model(
        model=mlflow_setup["model"],
        run_name="run",
        experiment_name=mlflow_setup["experiment"],
        params=super_params,
        model_name="test",
    )

    run = mlflow.get_run(run_id)

    assert run.data.params == super_params


def test_register_model_with_artifacts(mlflow_setup):
    dgw: MLFlowDataGateway = mlflow_setup["dgw"]
    hg = {"HolyGraal"}
    artifact_path = str(mlflow_setup["path"] / "hg.pkl")
    pickle.dump(hg, open(artifact_path, "wb"))

    artifacts = {"nice_glass": artifact_path}

    run_id = dgw.register_model(
        model=mlflow_setup["model"],
        run_name="run",
        experiment_name=mlflow_setup["experiment"],
        artifacts=artifacts,
        model_name="test",
    )

    run = mlflow.get_run(run_id)
    mlflow_artifacts = eval(run.data.tags["mlflow.log-model.history"])[0]["flavors"][
        "python_function"
    ]["artifacts"]

    assert "nice_glass" in mlflow_artifacts
    assert Path(f"{run.info.artifact_uri}/model/artifacts/hg.pkl").exists()


def test_register_model_with_metrics(mlflow_setup):
    dgw: MLFlowDataGateway = mlflow_setup["dgw"]
    metrics = {"power_level": 9001.0}

    run_id = dgw.register_model(
        model=Predictor(),
        model_name="test",
        experiment_name=mlflow_setup["experiment"],
        run_name="run",
        metrics=metrics,
    )

    run = mlflow.get_run(run_id)
    assert run.data.metrics == metrics
