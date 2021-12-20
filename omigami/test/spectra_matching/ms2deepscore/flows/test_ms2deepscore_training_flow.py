import os
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pytest
from drfs.filesystems import get_fs

from omigami.config import GNPS_URIS
from omigami.spectra_matching.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.storage import FSDataGateway
from omigami.test.spectra_matching.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


def test_training_flow(flow_config, mock_ms2ds_deploy_model_task):
    mock_data_gtw = MagicMock(spec=FSDataGateway)
    mock_spectrum_dgw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    flow_name = "test-flow"
    expected_tasks = {
        "DownloadData",
        "CreateChunks",
        "CleanRawSpectra",
        "CacheCleanedSpectra",
        "ProcessSpectrum",
        "CalculateTanimotoScore",
        "TrainModel",
        "RegisterModel",
        "CreateSpectrumIDsChunks",
        "MakeEmbeddings",
        "DeployModel",
    }

    flow_parameters = TrainingFlowParameters(
        fs_dgw=mock_data_gtw,
        spectrum_dgw=mock_spectrum_dgw,
        source_uri="source_uri",
        dataset_directory="datasets",
        ion_mode="positive",
        chunk_size=150000,
        overwrite_model=True,
        scores_output_path="some-path",
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        model_output_path="some-path",
        spectrum_binner_output_path="some-path",
        project_name="test",
        mlflow_output_directory="model-output",
        train_ratio=0.6,
        validation_ratio=0.3,
        test_ratio=0.1,
        spectrum_ids_chunk_size=10,
    )

    flow = build_training_flow(
        flow_name=flow_name,
        flow_config=flow_config,
        flow_parameters=flow_parameters,
        deploy_model=True,
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
    tmpdir,
    flow_config,
    mock_default_config,
    clean_chunk_files,
    redis_full_setup,
    small_model_params,
    mock_ms2ds_deploy_model_task,
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    data_gtw = MS2DeepScoreFSDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    flow_params = TrainingFlowParameters(
        fs_dgw=data_gtw,
        spectrum_dgw=spectrum_dgw,
        source_uri=GNPS_URIS["small"],
        dataset_directory=ASSETS_DIR,
        dataset_name="SMALL_GNPS.json",
        chunk_size=150000,
        ion_mode="positive",
        overwrite_model=True,
        scores_output_path=str(tmpdir / "tanimoto_scores.pkl"),
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        spectrum_binner_output_path=str(tmpdir / "spectrum_binner.pkl"),
        model_output_path=str(tmpdir / "model.hdf5"),
        epochs=1,
        project_name="test",
        mlflow_output_directory=f"{tmpdir}/model-output",
        train_ratio=0.8,
        validation_ratio=0.2,
        test_ratio=0.2,
        spectrum_ids_chunk_size=100,
    )

    flow = build_training_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        deploy_model=True,
    )

    flow_run = flow.run()
    download_task = flow.get_tasks("DownloadData")[0]
    register_task = flow.get_tasks("RegisterModel")[0]

    assert flow_run.is_successful()
    flow_run.result[download_task].is_cached()
    assert len(fs.ls(ASSETS_DIR / "raw/positive")) == 4
    assert fs.exists(ASSETS_DIR / "raw/positive/raw_chunk_paths.pickle")
    assert fs.exists(tmpdir / "tanimoto_scores.pkl")
    assert fs.exists(tmpdir / "model.hdf5")

    run_id = flow_run.result[register_task].result
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    model_uri = f"{artifact_uri}/model/"
    assert Path(model_uri).exists()
