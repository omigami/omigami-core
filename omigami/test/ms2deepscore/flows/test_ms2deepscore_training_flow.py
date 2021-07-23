import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.config import SOURCE_URI_PARTIAL_GNPS_500_SPECTRA
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
    ModelGeneralParameters,
)
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectrum_cleaner import SpectrumCleaner
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
    mock_spectrum_dgw = MagicMock(spec=MS2DeepScoreRedisSpectrumDataGateway)
    mock_cleaner = MagicMock(spec=SpectrumCleaner)
    flow_name = "test-flow"
    expected_tasks = {
        "DownloadData",
        "CreateChunks",
        "SaveRawSpectra",
        "ProcessSpectrum",
        "CalculateTanimotoScore",
        "TrainModel",
    }

    flow_parameters = TrainingFlowParameters(
        input_dgw=mock_input_dgw,
        spectrum_dgw=mock_spectrum_dgw,
        spectrum_cleaner=mock_cleaner,
        source_uri="source_uri",
        output_dir="datasets",
        dataset_id="dataset-id",
        ion_mode="positive",
        chunk_size=150000,
        overwrite_model=True,
        overwrite_all_spectra=False,
        scores_output_path="some-path",
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        model_output_path="some-path",
        spectrum_binner_output_path="some-path",
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


def test_run_training_flow(
    tmpdir, flow_config, mock_default_config, clean_chunk_files, redis_full_setup
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

    input_dgw = FSInputDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()
    spectrum_cleaner = SpectrumCleaner()

    flow_params = TrainingFlowParameters(
        input_dgw=input_dgw,
        spectrum_dgw=spectrum_dgw,
        spectrum_cleaner=spectrum_cleaner,
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
        output_dir=ASSETS_DIR.parent,
        dataset_id=ASSETS_DIR.name,
        dataset_name="SMALL_GNPS_500_spectra.json",
        chunk_size=150000,
        ion_mode="positive",
        overwrite_model=True,
        overwrite_all_spectra=True,
        scores_output_path=str(tmpdir / "tanimoto_scores.pkl"),
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        spectrum_binner_output_path=str(tmpdir / "spectrum_binner.pkl"),
        model_output_path=str(tmpdir / "model.hdf5"),
        dataset_checkpoint_name="spectrum_ids_500.pkl",
        epochs=10,
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
    assert len(fs.ls(ASSETS_DIR / "chunks/positive")) == 18
    assert fs.exists(ASSETS_DIR / "chunks/positive/chunk_paths.pickle")
    assert fs.exists(tmpdir / "tanimoto_scores.pkl")
    assert fs.exists(tmpdir / "model.hdf5")
