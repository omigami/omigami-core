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
from omigami.gateways.data_gateway import DataGateway
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.flows.pretrained_flow import (
    build_pretrained_flow,
    PretrainedFlowParameters,
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


def test_pretrained_flow(flow_config, spectra_stored):
    mock_data_gtw = MagicMock(spec=DataGateway)
    spectrum_dgw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)

    expected_tasks = {
        "ProcessSpectrum",
        "RegisterModel",
    }
    flow_params = PretrainedFlowParameters(
        model_uri="some model",
        data_gtw=mock_data_gtw,
        spectrum_dgw=spectrum_dgw,
        spectrum_binner_output_path="some path",
    )

    flow = build_pretrained_flow(
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
def test_run_pretrained_flow(
    tmpdir,
    flow_config,
    mock_default_config,
    spectra_stored,
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    data_gtw = FSDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    flow_params = PretrainedFlowParameters(
        data_gtw=data_gtw,
        spectrum_dgw=spectrum_dgw,
        model_uri=str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        ),
        spectrum_binner_output_path=str(tmpdir / "spectrum_binner.pkl"),
    )

    flow = build_pretrained_flow(
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
