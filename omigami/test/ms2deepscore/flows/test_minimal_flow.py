import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs
from omigami.data_gateway import InputDataGateway
from omigami.ms2deepscore.config import MS2DEEPSCORE_MODEL_URI

from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.ms2deepscore.flows.minimal_flow import (
    build_minimal_flow,
    MinimalFlowParameters,
)
from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway
from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


@pytest.fixture
def flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-hermione-XXII",
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="0",
    )
    return flow_config


def test_minimal_flow(flow_config):
    mock_input_dgw = MagicMock(spec=InputDataGateway)
    expected_tasks = {
        "DownloadPreTrainedModel",
        "RegisterModel",
    }
    flow_params = MinimalFlowParameters(
        input_dgw=mock_input_dgw,
        model_output_dir="model-output",
        model_uri="model_uri",
    )

    flow = build_minimal_flow(
        project_name="test",
        flow_name="test-flow",
        flow_config=flow_config,
        flow_parameters=flow_params,
        model_output_dir="model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )

    assert flow
    assert len(flow.tasks) == len(expected_tasks)
    assert flow.name == "test-flow"

    task_names = {t.name for t in flow.tasks}
    assert task_names == expected_tasks


def test_run_minimal_flow(
    tmpdir,
    flow_config,
    mock_default_config,
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()

    flow_params = MinimalFlowParameters(
        input_dgw=input_dgw,
        output_dir=ASSETS_DIR,
        model_uri=MS2DEEPSCORE_MODEL_URI,
    )

    flow = build_minimal_flow(
        project_name="test",
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        model_output_dir=f"{tmpdir}/model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadPreTrainedModel")

    assert results.is_successful()
    results.result[d].is_cached()
    assert "model" in os.listdir(tmpdir / "model-output")
