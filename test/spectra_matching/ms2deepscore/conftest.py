import pickle
from typing import List

import mlflow
import pandas as pd
import pytest
from mlflow.entities import Experiment
from ms2deepscore.models import load_model
from pytest_redis import factories

import omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer
from omigami.config import GNPS_URIS, EMBEDDING_HASHES
from omigami.spectra_matching.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.spectra_matching.ms2deepscore.embedding import MS2DeepScoreEmbedding
from omigami.spectra_matching.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.spectra_matching.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.tasks import (
    RegisterModelParameters,
    TrainModelParameters,
    RegisterModel,
    MakeEmbeddings,
)
from omigami.spectra_matching.storage import FSDataGateway
from test.spectra_matching.conftest import ASSETS_DIR, TEST_TASK_CONFIG, CACHE_DIR
from test.spectra_matching.tasks import DummyTask

redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture
def positive_spectra_data(common_cleaned_data):
    spectra = [
        data for data in common_cleaned_data if data.get("ionmode") == "positive"
    ]
    return spectra


@pytest.fixture
def positive_spectra(positive_spectra_data):
    spectra = SpectrumProcessor().process_spectra(positive_spectra_data)
    return spectra


@pytest.fixture
def ms2ds_payload(raw_spectra):
    spectra = [data for data in raw_spectra if data["Ion_Mode"] == "Positive"]
    payload = {
        "data": {
            "ndarray": {
                "parameters": {
                    "n_best_spectra": 2,
                },
                "data": [
                    {
                        "peaks_json": spectra[0]["peaks_json"],
                        "Precursor_MZ": spectra[0]["Precursor_MZ"],
                    },
                    {
                        "peaks_json": spectra[1]["peaks_json"],
                        "Precursor_MZ": spectra[1]["Precursor_MZ"],
                    },
                ],
            }
        }
    }
    return payload


@pytest.fixture
def siamese_model_path(
    small_model_params, mock_default_config, ms2ds_build_test_model_flow
):
    """Trains and caches a MS2DS model. To retrain. delete cached model."""
    path = CACHE_DIR / "ms2deep_score.hdf5"

    if not path.exists():
        path.parent.mkdir(exist_ok=True)
        res = ms2ds_build_test_model_flow.run()
        assert res.is_successful()

    return str(path)


@pytest.fixture
def siamese_model(siamese_model_path):
    ms2deepscore_model = load_model(siamese_model_path)
    return ms2deepscore_model


@pytest.fixture
def ms2deepscore_predictor(siamese_model):
    ms2deepscore_predictor = MS2DeepScorePredictor(ion_mode="positive", run_id="2")
    ms2deepscore_predictor.model = siamese_model
    return ms2deepscore_predictor


@pytest.fixture
def tanimoto_scores_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "tanimoto_scores.pkl")


@pytest.fixture
def tanimoto_scores(tanimoto_scores_path):
    tanimoto_score = pd.read_pickle(tanimoto_scores_path, compression="gzip")
    return tanimoto_score


@pytest.fixture
def binned_spectra_to_train():
    path = str(ASSETS_DIR / "ms2deepscore" / "to_train" / "binned_spectra.pkl")
    with open(path, "rb") as f:
        binned_spectra = pickle.load(f)
    return binned_spectra


@pytest.fixture
def binned_spectra_to_train_path():
    return ASSETS_DIR / "ms2deepscore" / "to_train" / "binned_spectra.pkl"

@pytest.fixture
def small_model_params(monkeypatch):
    smaller_params = {
        "batch_size": 2,
        "learning_rate": 0.001,
        "layer_base_dims": (300, 200, 100),
        "embedding_dim": 100,
        "dropout_rate": 0.2,
    }

    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer,
        "SIAMESE_MODEL_PARAMS",
        smaller_params,
    )


@pytest.fixture
def mock_ms2ds_deploy_model_task(monkeypatch):

    import omigami.spectra_matching.ms2deepscore.flows.deploy_model
    import omigami.spectra_matching.ms2deepscore.flows.training_flow

    class DeployModel(DummyTask):
        pass

    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.deploy_model,
        "DeployModel",
        DeployModel,
    )
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.training_flow,
        "DeployModel",
        DeployModel,
    )


@pytest.fixture
def ms2ds_build_test_model_flow(tmpdir, flow_config, monkeypatch, clean_chunk_files):
    """Used to generate local asset ms2deep_score.hd5 for multiple tests."""
    import omigami.spectra_matching.ms2deepscore.flows.training_flow

    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.training_flow,
        "DownloadData",
        DummyTask,
    )
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.training_flow,
        "MakeEmbeddings",
        DummyTask,
    )
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.training_flow,
        "RegisterModel",
        DummyTask,
    )
    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.training_flow,
        "DeployModel",
        DummyTask,
    )

    data_gtw = MS2DeepScoreFSDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    flow_params = TrainingFlowParameters(
        fs_dgw=data_gtw,
        spectrum_dgw=spectrum_dgw,
        source_uri=GNPS_URIS["small_500"],
        # the three parameters below are for using cached assets instead of downloading
        dataset_directory=str(ASSETS_DIR),
        dataset_name="SMALL_GNPS_500_spectra.json",
        chunk_size=150000,
        ion_mode="positive",
        # we use everything but the model path as tmpdir. We only want the model from this script
        scores_output_path=str(tmpdir / "tanimoto_scores.pkl"),
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        spectrum_binner_output_path=str(tmpdir / "spectrum_binner.pkl"),
        model_output_path=str(CACHE_DIR / "ms2deep_score.hdf5"),
        epochs=5,
        project_name="test",
        mlflow_output_directory=f"{tmpdir}/model-output",
        train_ratio=0.6,
        validation_ratio=0.2,
        test_ratio=0.2,
        spectrum_ids_chunk_size=100,
    )

    flow = build_training_flow(
        flow_config=flow_config,
        flow_name="test-flow",
        flow_parameters=flow_params,
    )

    return flow


@pytest.fixture
def registered_ms2ds_model(siamese_model_path):
    """
    Registers a model in a local mlflow. Skips if already registered.
    For resetting cache, delete `assets/cache/mlflow`
    """
    mlflow_path = CACHE_DIR / "mlflow/mlflow.sqlite"
    mlflow_path.parent.mkdir(exist_ok=True)
    mlflow_server = "sqlite:///" + str(mlflow_path)
    experiment_name = "test_fixture"

    if mlflow_path.exists():
        client = mlflow.tracking.MlflowClient(tracking_uri=mlflow_server)
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            model_data = _load_predictor_from_mlflow(client, experiment)
            model_data["model_path"] = siamese_model_path
            model_data["mlflow_uri"] = mlflow_server
            return model_data

    params = RegisterModelParameters(
        experiment_name, mlflow_server, str(mlflow_path.parent), "positive"
    )
    train_params = TrainModelParameters("path", "positive", "path")
    register_task = RegisterModel(params, train_params)
    model_run_id = register_task.run(
        {"ms2deepscore_model_path": siamese_model_path, "validation_loss": 0.5}
    )
    run = mlflow.get_run(model_run_id)
    model_uri = run.info.artifact_uri + "/model"
    predictor: MS2DeepScorePredictor = mlflow.pyfunc.load_model(
        model_uri
    )._model_impl.python_model

    return {
        "run_id": model_run_id,
        "mlflow_uri": mlflow_server,
        "predictor": predictor,
        "model_path": siamese_model_path,
    }


def _load_predictor_from_mlflow(
    client: mlflow.tracking.MlflowClient, experiment: Experiment
) -> dict:
    """Skips registering model if there is one already present (i.e. if the fixture
    has been executed before"""
    run = client.list_run_infos(experiment.experiment_id)[0]
    model_uri = run.artifact_uri + "/model"
    predictor: MS2DeepScorePredictor = mlflow.pyfunc.load_model(
        model_uri
    )._model_impl.python_model
    model_run_id = run.run_id
    model_data = {
        "run_id": model_run_id,
        "predictor": predictor,
    }
    return model_data


@pytest.fixture
def ms2ds_saved_embeddings(
    registered_ms2ds_model, binned_spectra_stored, spectra_stored, redis_db
):
    """
    Generates embeddings using registered ms2ds model. Uses cache instead, if available.
    """
    fs_dgw = MS2DeepScoreFSDataGateway()
    spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()
    cache_path = CACHE_DIR / "ms2ds_embeddings.pickle"

    if cache_path.exists():
        embeddings = _get_cached_embeddings(cache_path, fs_dgw, redis_db)
        return embeddings

    task = MakeEmbeddings(spectrum_dgw, fs_dgw, "positive", **TEST_TASK_CONFIG)
    embedding_ids = task.run(
        {
            "ms2deepscore_model_path": registered_ms2ds_model["model_path"],
            "validation_loss": 0.5,
        },
        registered_ms2ds_model["run_id"],
        spectrum_ids=spectrum_dgw.list_spectrum_ids(),
    )

    embeddings = spectrum_dgw.read_embeddings("positive", embedding_ids)
    fs_dgw.serialize_to_file(cache_path, embeddings)

    return embeddings


def _get_cached_embeddings(
    cache_path: str, fs_dgw: FSDataGateway, redis_db
) -> List[MS2DeepScoreEmbedding]:
    """Loads embeddings from cache and saves to redis"""
    embeddings = fs_dgw.read_from_file(cache_path)
    project = "ms2deepscore"
    ion_mode = "positive"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{project}_{ion_mode}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()
    return embeddings
