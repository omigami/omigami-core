import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs
from prefect import Flow
from prefect.executors import LocalDaskExecutor
from prefect.storage import S3

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.data_gateway import SpectrumDataGateway
from omigami.flows.config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
    SpectrumIonMode,
)
from omigami.flows.training_flow import build_training_flow
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.tasks import (
    DeployModel,
    DownloadParameters,
    ProcessSpectrumParameters,
    TrainModelParameters,
)
from omigami.test.conftest import ASSETS_DIR, TEST_TASK_CONFIG

os.chdir(Path(__file__).parents[3])


@pytest.fixture
def flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="0",
    )
    return flow_config


def test_training_flow(flow_config):
    mock_spectrum_dgw = MagicMock(spec=SpectrumDataGateway)
    mock_input_dgw = MagicMock(spec=FSInputDataGateway)
    expected_tasks = {
        "CreateChunks",
        "DownloadData",
        "ProcessSpectrum",
        "MakeEmbeddings",
        "RegisterModel",
        "TrainModel",
    }

    flow = build_training_flow(
        project_name="test",
        input_dgw=mock_input_dgw,
        download_params=DownloadParameters("source_uri", "datasets", "dataset-id"),
        process_params=ProcessSpectrumParameters(mock_spectrum_dgw, 2, False),
        train_params=TrainModelParameters(mock_spectrum_dgw, 25, 500),
        model_output_dir="model-output",
        mlflow_server="mlflow-server",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=flow_config,
        redis_db="0",
        deploy_model=False,
    )

    assert flow
    assert len(flow.tasks) == len(expected_tasks)
    assert flow.name == "spec2vec-training-flow"

    task_names = {t.name for t in flow.tasks}
    assert task_names == expected_tasks


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_run_training_flow(tmpdir, flow_config, mock_default_config, clean_chunk_files):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()
    download_parameters = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, ASSETS_DIR, "SMALL_GNPS.json"
    )
    spectrum_dgw = RedisSpectrumDataGateway()
    process_parameters = ProcessSpectrumParameters(spectrum_dgw, 1, False)
    train_params = TrainModelParameters(spectrum_dgw, 3, 200)

    flow = build_training_flow(
        project_name="test",
        input_dgw=input_dgw,
        download_params=download_parameters,
        process_params=process_parameters,
        model_output_dir=f"{tmpdir}/model-output",
        mlflow_server="mlflow-server",
        train_params=train_params,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        flow_config=flow_config,
        chunk_size=150000,
        ion_mode=SpectrumIonMode.POSITIVE.value,
        redis_db="0",
        deploy_model=False,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadData")

    assert results.is_successful()
    results.result[d].is_cached()
    assert "model" in os.listdir(tmpdir / "model-output")
    assert len(fs.ls(ASSETS_DIR / "chunks")) == 6
    assert fs.exists(ASSETS_DIR / "chunk_paths.pickle")


@pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model():
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    }

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        DeployModel(redis_db="0", environment="dev", **TEST_TASK_CONFIG)(
            registered_model={
                "model_uri": "s3://omigami-dev/spec2vec/mlflow/ece0dbc5aba84322ae3a2cc6ae97212b/artifacts/model/"
            }
        )

    res = deploy.run()

    assert res.is_successful()
