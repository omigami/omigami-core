import os
from pathlib import Path

import pytest
from drfs.filesystems import get_fs
from prefect import task

import spec2vec_mlops.flows.training_flow
from spec2vec_mlops import config
from spec2vec_mlops.flows.training_flow import build_training_flow

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[3])


@task()
def mock_task(a=None, b=None, c=None, **kwargs):
    pass


def test_training_flow():
    expected_tasks = {
        "DownloadData",
        "LoadData",
        "case(True)",
        "check_condition",
        "clean_data_task",
        "deploy_model_task",
        "make_embeddings_task",
        "register_model_task",
        "train_model_task",
    }

    flow = build_training_flow(
        project_name="test",
        source_uri="source_uri",
        dataset_dir="datasets",
        dataset_id="dataset-id",
        model_output_dir="model-output",
        seldon_deployment_path="seldon-path",
        n_decimals=2,
        mlflow_server="mlflow-server",
        iterations=25,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=None,
    )

    assert flow
    assert len(flow.tasks) == 9
    assert flow.name == "spec2vec-training-flow"

    task_names = {t.name for t in flow.tasks}
    assert task_names == expected_tasks

    download_task = list(flow.tasks)[0]
    assert flow.constants[download_task] == {
        "input_uri": "source_uri",
        "output_dir": "datasets",
        "dataset_id": "dataset-id",
    }


@pytest.fixture()
def mock_seldom_deployment(monkeypatch):
    monkeypatch.setattr(
        spec2vec_mlops.flows.training_flow, "deploy_model_task", mock_task
    )


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_run_training_flow(mock_seldom_deployment, tmpdir):
    current_dir = Path.cwd()
    fs = get_fs(str(current_dir))
    flow = build_training_flow(
        project_name="test",
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        # we dont use tmpdir here to avoid downloading it every time
        dataset_dir=f"datasets",
        dataset_id="dataset-id",
        model_output_dir=f"{tmpdir}/model-output",
        seldon_deployment_path="seldon-path",
        n_decimals=2,
        mlflow_server="mlflow-server",
        iterations=25,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=None,
    )

    results = flow.run()

    assert results.is_successful()
    assert len(fs.ls(current_dir / "datasets")) == 1
    assert len(fs.ls(tmpdir / "model-output")) == 1
