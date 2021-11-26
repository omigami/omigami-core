import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.spectra_matching.gateways import RedisSpectrumDataGateway
from omigami.spectra_matching.gateways.fs_data_gateway import FSDataGateway
from omigami.spectra_matching.spec2vec.config import PROJECT_NAME
from omigami.spectra_matching.spec2vec.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spectra_matching.spec2vec.gateways.redis_spectrum_document import (
    RedisSpectrumDocumentDataGateway,
)
from omigami.spectra_matching.spec2vec.gateways.spectrum_document import (
    SpectrumDocumentDataGateway,
)
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


def test_training_flow(flow_config):
    mock_spectrum_dgw = MagicMock(spec=RedisSpectrumDataGateway)
    mock_data_gtw = MagicMock(spec=FSDataGateway)
    mock_document_dgw = MagicMock(spec=SpectrumDocumentDataGateway)
    mock_cleaner = MagicMock(spec=SpectrumCleaner)
    expected_tasks = {
        "DownloadData",
        "CreateChunks",
        "SaveRawSpectra",
        "ProcessSpectrum",
        "MakeEmbeddings",
        "RegisterModel",
        "TrainModel",
    }
    flow_params = TrainingFlowParameters(
        spectrum_dgw=mock_spectrum_dgw,
        data_gtw=mock_data_gtw,
        document_dgw=mock_document_dgw,
        spectrum_cleaner=mock_cleaner,
        source_uri="source_uri",
        dataset_directory="datasets",
        dataset_id="dataset-id",
        chunk_size=150000,
        ion_mode="positive",
        n_decimals=2,
        overwrite_model=True,
        overwrite_all_spectra=False,
        iterations=25,
        window=500,
        experiment_name="test",
        mlflow_output_directory="model-output",
        documents_save_directory="documents",
        mlflow_server="mlflow-server",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
    )

    flow = build_training_flow(
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
def test_run_training_flow(
    tmpdir,
    flow_config,
    mock_default_config,
    clean_chunk_files,
    spec2vec_redis_setup,
):
    # remove mlflow models from previous runs
    mlflow_root = tmpdir / "test-mlflow"
    if mlflow_root.exists():
        shutil.rmtree(mlflow_root)
    fs = get_fs(mlflow_root)
    ion_mode = "positive"

    spectrum_dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    data_gtw = FSDataGateway()
    document_dgw = RedisSpectrumDocumentDataGateway()
    spectrum_cleaner = SpectrumCleaner()
    flow_params = TrainingFlowParameters(
        spectrum_dgw=spectrum_dgw,
        data_gtw=data_gtw,
        document_dgw=document_dgw,
        spectrum_cleaner=spectrum_cleaner,
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        dataset_directory=ASSETS_DIR.parent,
        dataset_id=ASSETS_DIR.name,
        dataset_name="SMALL_GNPS.json",
        chunk_size=150000,
        ion_mode="positive",
        n_decimals=1,
        overwrite_model=True,
        overwrite_all_spectra=True,
        iterations=3,
        window=200,
        experiment_name="test",
        mlflow_output_directory=str(mlflow_root),
        documents_save_directory=tmpdir / f"documents/{ion_mode}",
        mlflow_server=str(mlflow_root),
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        model_name=None,
    )

    flow = build_training_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        deploy_model=False,
    )

    flow_run = flow.run()
    download_task = flow.get_tasks("DownloadData")[0]
    register_task = flow.get_tasks("RegisterModel")[0]

    assert flow_run.is_successful()
    flow_run.result[download_task].is_cached()
    model_uri = flow_run.result[register_task].result["model_uri"]
    assert Path(model_uri).exists()
    assert len(fs.ls(ASSETS_DIR / "chunks/positive")) == 4
    assert fs.exists(ASSETS_DIR / "chunks/positive/chunk_paths.pickle")
