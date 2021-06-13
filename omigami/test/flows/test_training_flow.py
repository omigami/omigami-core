import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.data_gateway import SpectrumDataGateway
from omigami.flows.config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.flows.training_flow import build_training_flow
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.tasks import (
    DownloadParameters,
    ProcessSpectrumParameters,
    TrainModelParameters,
)
from omigami.test.conftest import ASSETS_DIR

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
        download_parameters=DownloadParameters("source_uri", "datasets", "dataset-id"),
        processing_parameters=ProcessSpectrumParameters(mock_spectrum_dgw, 2, False),
        training_parameters=TrainModelParameters(mock_spectrum_dgw, 25, 500),
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
def test_run_training_flow(
    tmpdir, flow_config, mock_default_config, clean_chunk_files, redis_full_setup
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()
    download_parameters = DownloadParameters(
        SOURCE_URI_PARTIAL_GNPS, ASSETS_DIR, "SMALL_GNPS.json"
    )
    spectrum_dgw = RedisSpectrumDataGateway()
    process_parameters = ProcessSpectrumParameters(spectrum_dgw, 1, True)
    train_params = TrainModelParameters(spectrum_dgw, 3, 200)

    flow = build_training_flow(
        project_name="test",
        input_dgw=input_dgw,
        download_parameters=download_parameters,
        processing_parameters=process_parameters,
        model_output_dir=f"{tmpdir}/model-output",
        mlflow_server="mlflow-server",
        training_parameters=train_params,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        flow_config=flow_config,
        chunk_size=150000,
        ion_mode="positive",
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
