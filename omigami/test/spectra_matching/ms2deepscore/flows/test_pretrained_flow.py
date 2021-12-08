import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.spectra_matching.ms2deepscore.flows.pretrained_flow import (
    build_pretrained_flow,
    PretrainedFlowParameters,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.storage import DataGateway, FSDataGateway
from omigami.test.spectra_matching.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


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
        project_name="test",
        mlflow_output_dir="model-output",
    )

    flow = build_pretrained_flow(
        flow_name="test-flow",
        flow_config=flow_config,
        flow_parameters=flow_params,
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
@pytest.mark.xfail
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
        project_name="test",
        mlflow_output_dir=f"{tmpdir}/mlflow-model-output",
    )

    flow = build_pretrained_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        deploy_model=False,
    )

    results = flow.run()
    register_task = flow.get_tasks("RegisterModel")[0]

    assert results.is_successful()
    model_uri = results.result[register_task].result["model_uri"]
    assert Path(model_uri).exists()
