import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.data_gateway import InputDataGateway
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.flows.minimal_flow import (
    build_minimal_flow,
    MinimalFlowParameters,
)
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
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


def test_minimal_flow(flow_config, spectra_stored):
    mock_input_dgw = MagicMock(spec=InputDataGateway)
    spectrum_dgw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)

    expected_tasks = {
        "CreateSpectrumIDsChunks",
        "ProcessSpectrum",
        "RegisterModel",
    }
    flow_params = MinimalFlowParameters(
        model_uri="some model",
        input_dgw=mock_input_dgw,
        spectrum_dgw=spectrum_dgw,
        spectrum_ids_chunk_size=10,
    )

    flow = build_minimal_flow(
        project_name="test",
        flow_name="test-flow",
        flow_config=flow_config,
        flow_parameters=flow_params,
        mlflow_output_dir="model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )

    assert flow
    assert len(flow.tasks) == len(expected_tasks)
    assert flow.name == "test-flow"

    task_names = {t.name for t in flow.tasks}
    assert task_names == expected_tasks


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.skipif(
    not os.path.exists(
        str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        )
    ),
    reason="MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 is git ignored. Please "
    "download it from https://zenodo.org/record/4699356#.YNyD-2ZKhcA",
)
def test_run_minimal_flow(
    tmpdir,
    flow_config,
    mock_default_config,
    spectra_stored,
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    flow_params = MinimalFlowParameters(
        input_dgw=input_dgw,
        spectrum_dgw=spectrum_dgw,
        model_uri=str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        ),
        spectrum_ids_chunk_size=10,
    )

    flow = build_minimal_flow(
        project_name="test",
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        mlflow_output_dir=f"{tmpdir}/mlflow-model-output",
        mlflow_server="mlflow-server",
        deploy_model=False,
    )

    results = flow.run()

    assert results.is_successful()
    assert "model" in os.listdir(tmpdir / "mlflow-model-output")
