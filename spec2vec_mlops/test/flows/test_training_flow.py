import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs import DRPath
from drfs.filesystems import get_fs
from prefect import task, Flow
from prefect.executors import LocalDaskExecutor
from prefect.storage import S3

import spec2vec_mlops.flows.training_flow
from spec2vec_mlops import config
from spec2vec_mlops.deployment import OUTPUT_DIR, DATASET_DIR, MODEL_DIR, MLFLOW_SERVER
from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.data_gateway import SpectrumDataGateway
from spec2vec_mlops.tasks import deploy_model_task
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrumParameters
from spec2vec_mlops.test.conftest import ASSETS_DIR

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[3])


@task()
def mock_task(a=None, b=None, c=None, **kwargs):
    pass


def test_training_flow():
    mock_spectrum_dgw = MagicMock(spec=SpectrumDataGateway)
    mock_input_dgw = MagicMock(spec=FSInputDataGateway)
    expected_tasks = {
        "CreateChunks",
        "DownloadData",
        "ProcessSpectrum",
        "case(True)",
        "check_condition",
        "deploy_model_task",
        "MakeEmbeddings",
        "register_model_task",
        "train_model_task",
    }

    flow = build_training_flow(
        project_name="test",
        download_params=DownloadParameters(
            "source_uri", "datasets", "dataset-id", mock_input_dgw
        ),
        process_params=ProcessSpectrumParameters(
            mock_spectrum_dgw, mock_input_dgw, 2, False
        ),
        model_output_dir="model-output",
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


@pytest.fixture()
def mock_seldon_deployment(monkeypatch):
    monkeypatch.setattr(
        spec2vec_mlops.flows.training_flow, "deploy_model_task", mock_task
    )


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_run_training_flow(mock_seldon_deployment, tmpdir):
    # remove results from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()
    download_parameters = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, ASSETS_DIR, "SMALL_GNPS.json", input_dgw
    )
    spectrum_dgw = RedisSpectrumDataGateway()
    process_parameters = ProcessSpectrumParameters(spectrum_dgw, input_dgw, 2, True)

    flow = build_training_flow(
        project_name="test",
        download_params=download_parameters,
        process_params=process_parameters,
        model_output_dir=f"{tmpdir}/model-output",
        mlflow_server="mlflow-server",
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=None,
        chunk_size=10,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadData")

    assert results.is_successful()
    results.result[d].is_cached()
    assert len(fs.ls(tmpdir / "model-output")) == 1


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.skip("Uses internet connection.")
def test_run_training_flow_with_s3_data(mock_seldon_deployment):
    fs = get_fs(OUTPUT_DIR)

    input_dgw = FSInputDataGateway()
    download_parameters = DownloadParameters(
        "fake_10k_dataset_uri",
        OUTPUT_DIR,
        DATASET_DIR["10k"] + "gnps.json",
        input_dgw,
        DATASET_DIR["10k"] + "spectrum_ids.pkl",
    )
    spectrum_dgw = RedisSpectrumDataGateway()
    process_parameters = ProcessSpectrumParameters(spectrum_dgw, input_dgw, 2, False)
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    }

    flow = build_training_flow(
        project_name="test-project",
        download_params=download_parameters,
        process_params=process_parameters,
        model_output_dir=str(DRPath(f"{MODEL_DIR}/tests")),
        mlflow_server=MLFLOW_SERVER,
        iterations=5,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=FLOW_CONFIG,
        chunk_size=1000,
        flow_name="test-flow",
    )

    results = flow.run()

    assert results.is_successful()


@pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model():
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    }

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        deploy_model_task(
            {
                "model_uri": "s3://dr-prefect/spec2vec-training-flow/mlflow/tests/e06d4ef7116e4bc78b76fc867fff29dc/artifacts/model/"
            }
        )

    res = deploy.run()
