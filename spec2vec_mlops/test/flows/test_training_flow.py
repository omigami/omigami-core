import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs
from prefect import Flow
from prefect.executors import LocalDaskExecutor
from prefect.storage import S3

from spec2vec_mlops.config import default_configs
from spec2vec_mlops.flows.config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.data_gateway import SpectrumDataGateway
from spec2vec_mlops.tasks import DeployModelTask
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrumParameters
from spec2vec_mlops.test.conftest import ASSETS_DIR

SOURCE_URI_PARTIAL_GNPS = default_configs["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[3])


@pytest.fixture
def flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="2",
    )
    return flow_config


def test_training_flow(flow_config):
    mock_spectrum_dgw = MagicMock(spec=SpectrumDataGateway)
    mock_input_dgw = MagicMock(spec=FSInputDataGateway)
    expected_tasks = {
        "CreateChunks",
        "DownloadData",
        "ProcessSpectrum",
        "case(True)",
        "check_condition",
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
def test_run_training_flow(tmpdir, flow_config):
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
        flow_config=flow_config,
        chunk_size=10,
        redis_db="0",
        deploy_model=False,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadData")

    assert results.is_successful()
    results.result[d].is_cached()
    assert len(fs.ls(tmpdir / "model-output")) == 1


@pytest.mark.skip(reason="This test deploys a seldon model using a model URI.")
def test_deploy_seldon_model():
    FLOW_CONFIG = {
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    }

    with Flow("debugging-flow", **FLOW_CONFIG) as deploy:
        DeployModelTask(redis_db="0",)(
            registered_model={
                "model_uri": "s3://dr-prefect/spec2vec-training-flow/mlflow/tests/750c60ddb52544289db228a4af8a52e3/artifacts/model/"
            }
        )

    res = deploy.run()
