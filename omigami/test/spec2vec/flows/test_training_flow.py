import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways import RedisSpectrumDataGateway
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spec2vec.gateways.redis_spectrum_document import (
    RedisSpectrumDocumentDataGateway,
)
from omigami.spec2vec.gateways.spectrum_document import SpectrumDocumentDataGateway
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
        output_dir="datasets",
        dataset_id="dataset-id",
        chunk_size=150000,
        ion_mode="positive",
        n_decimals=2,
        overwrite_model=True,
        overwrite_all_spectra=False,
        iterations=25,
        window=500,
        project_name="test",
        model_output_dir="model-output",
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
    redis_full_setup,
):
    # remove mlflow models from previous runs
    fs = get_fs(ASSETS_DIR)
    ion_mode = "positive"
    _ = [fs.rm(p) for p in fs.ls(tmpdir / "model-output")]

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
        output_dir=ASSETS_DIR.parent,
        dataset_id=ASSETS_DIR.name,
        dataset_name="SMALL_GNPS.json",
        chunk_size=150000,
        ion_mode=ion_mode,
        n_decimals=1,
        overwrite_model=True,
        overwrite_all_spectra=True,
        iterations=3,
        window=200,
        project_name="test",
        model_output_dir=f"{tmpdir}/model-output",
        documents_save_directory=f"{tmpdir}/documents/{ion_mode}",
        mlflow_server="mlflow-server",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
    )

    flow = build_training_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
        deploy_model=False,
    )

    results = flow.run()
    (d,) = flow.get_tasks("DownloadData")

    assert results.is_successful()
    results.result[d].is_cached()
    assert "model" in os.listdir(tmpdir / "model-output")
    assert len(fs.ls(ASSETS_DIR / "chunks/positive")) == 4
    assert fs.exists(ASSETS_DIR / "chunks/positive/chunk_paths.pickle")
