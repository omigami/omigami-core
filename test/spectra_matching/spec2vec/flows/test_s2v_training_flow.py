import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock

import mlflow
import pytest
from drfs.filesystems import get_fs

from omigami.config import GNPS_URIS
from omigami.spectra_matching.spec2vec import SPEC2VEC_PROJECT_NAME
from omigami.spectra_matching.spec2vec.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from test.spectra_matching.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])


def test_training_flow(flow_config):
    mock_data_gtw = MagicMock(spec=FSDataGateway)
    expected_tasks = {
        "DownloadData",
        "CreateChunks",
        "CleanRawSpectra",
        "CreateDocuments",
        "RegisterModel",
        "TrainModel",
    }
    flow_params = TrainingFlowParameters(
        fs_dgw=mock_data_gtw,
        source_uri="source_uri",
        dataset_directory="datasets",
        chunk_size=150000,
        ion_mode="positive",
        n_decimals=2,
        iterations=25,
        window=500,
        experiment_name="test",
        mlflow_output_directory="model-output",
        documents_save_directory="documents",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
    )

    flow = build_training_flow(
        flow_name="test-flow",
        flow_config=flow_config,
        flow_parameters=flow_params,
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
    clean_chunk_files,
):
    # remove mlflow models from previous runs
    mlflow_root = tmpdir / "test-mlflow"
    if mlflow_root.exists():
        shutil.rmtree(mlflow_root)
    fs = get_fs(mlflow_root)
    ion_mode = "positive"

    spectrum_dgw = RedisSpectrumDataGateway(project=SPEC2VEC_PROJECT_NAME)
    data_gtw = FSDataGateway()
    flow_params = TrainingFlowParameters(
        fs_dgw=data_gtw,
        source_uri=GNPS_URIS["small"],
        dataset_directory=ASSETS_DIR,
        dataset_name="SMALL_GNPS.json",
        chunk_size=int(1e8),
        ion_mode="positive",
        n_decimals=1,
        iterations=3,
        window=200,
        experiment_name="test",
        mlflow_output_directory=str(mlflow_root),
        documents_save_directory=tmpdir / f"documents/{ion_mode}",
        intensity_weighting_power=0.5,
        allowed_missing_percentage=25,
        model_name=None,
    )

    flow = build_training_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
    )

    flow_run = flow.run()
    download_task = flow.get_tasks("DownloadData")[0]
    register_task = flow.get_tasks("RegisterModel")[0]

    assert flow_run.is_successful()
    flow_run.result[download_task].is_cached()
    assert len(fs.ls(ASSETS_DIR / "raw/positive")) == 2
    assert fs.exists(ASSETS_DIR / "raw/positive/raw_chunk_paths.pickle")

    run_id = flow_run.result[register_task].result
    artifact_uri = mlflow.get_run(run_id).info.artifact_uri
    model_uri = f"{artifact_uri}/model/"
    assert Path(model_uri).exists()
