import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.gateways.data_gateway import SpectrumDataGateway

from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
    ModelGeneralParameters
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
    mock_spectrum_dgw = MagicMock(spec=SpectrumDataGateway)
    flow_name = "test-flow"
    expected_tasks = {
        "DownloadData",
    }

    flow_parameters = TrainingFlowParameters(
        spectrum_dgw=mock_spectrum_dgw,
        source_uri="source_uri",
        output_dir="datasets",
        dataset_id="dataset-id",
        ion_mode="positive",
    )
    model_parameters = ModelGeneralParameters(
        model_output_dir="model-output",
        mlflow_server="mlflow-server",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        deploy_model=False,
    )
    flow = build_training_flow(
        project_name="test",
        flow_name=flow_name,
        flow_config=flow_config,
        flow_parameters=flow_parameters,
        model_parameters=model_parameters
    )

    assert flow
    assert len(flow.tasks) == len(expected_tasks)
    assert flow.name == flow_name

    task_names = {t.name for t in flow.tasks}
    assert task_names == expected_tasks


def test_run_training_flow(
        tmpdir, flow_config, mock_default_config, clean_chunk_files, redis_full_setup
):
    pass
