import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.gateways.data_gateway import SpectrumDataGateway

from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
    ModelGeneralParameters,
)

from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


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
    mock_input_dgw = MagicMock(spec=FSInputDataGateway)
    mock_spectrum_dgw = MagicMock(spec=SpectrumDataGateway)
    flow_name = "test-flow"
    expected_tasks = {
        "DownloadData",
    }

    flow_parameters = TrainingFlowParameters(
        input_dgw=mock_input_dgw,
        spectrum_dgw=mock_spectrum_dgw,
        source_uri="source_uri",
        output_dir="datasets",
        dataset_id="dataset-id",
        ion_mode="positive",
    )
    model_parameters = ModelGeneralParameters(
        model_output_dir="model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )
    flow = build_training_flow(
        project_name="test",
        flow_name=flow_name,
        flow_config=flow_config,
        flow_parameters=flow_parameters,
        model_parameters=model_parameters,
    )

    assert flow
    assert len(flow.tasks) == len(expected_tasks)
    assert flow.name == flow_name

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

    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    flow_params = TrainingFlowParameters(
        input_dgw=input_dgw,
        spectrum_dgw=spectrum_dgw,
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        output_dir=ASSETS_DIR.parent,
        dataset_id=ASSETS_DIR.name,
        dataset_name="SMALL_GNPS.json",
        ion_mode="positive",
    )

    model_parameters = ModelGeneralParameters(
        model_output_dir=f"{tmpdir}/model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )

    flow = build_training_flow(
        project_name="test",
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        model_parameters=model_parameters,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadData")

    assert results.is_successful()
    results.result[d].is_cached()
    # Model does not yet get created by flow
    # assert "model" in os.listdir(tmpdir / "model-output")
    # assert len(fs.ls(ASSETS_DIR / "chunks/positive")) == 4
    # assert fs.exists(ASSETS_DIR / "chunks/positive/chunk_paths.pickle")
