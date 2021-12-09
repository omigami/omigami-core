import pickle

import pandas as pd
import pytest
from ms2deepscore.models import load_model
from pytest_redis import factories

import omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer
from omigami.config import SOURCE_URI_PARTIAL_GNPS_500_SPECTRA
from omigami.spectra_matching.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.spectra_matching.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)
from omigami.spectra_matching.ms2deepscore.predictor import (
    MS2DeepScorePredictor,
    MS2DeepScoreSimilarityScoreCalculator,
)
from omigami.spectra_matching.ms2deepscore.storage import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.tasks.clean_raw_spectra import SpectrumCleaner
from omigami.test.spectra_matching.conftest import ASSETS_DIR
from omigami.test.spectra_matching.tasks import DummyTask

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
def ms2deepscore_payload(loaded_data):
    spectra = [data for data in loaded_data if data["Ion_Mode"] == "Positive"]
    payload = {
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
        "parameters": {"n_best_spectra": 2, "include_metadata": ["Compound_name"]},
    }
    return payload


@pytest.fixture()
def ms2deepscore_model_path():
    return str(ASSETS_DIR / "ms2deepscore_model.hdf5")


@pytest.fixture()
def ms2deepscore_embedding(ms2deepscore_model_path):
    model = load_model(ms2deepscore_model_path)
    return MS2DeepScoreSimilarityScoreCalculator(model)


@pytest.fixture()
def ms2deepscore_predictor(ms2deepscore_embedding):
    predictor = MS2DeepScorePredictor(ion_mode="positive", run_id="2")
    predictor.model = ms2deepscore_embedding
    return predictor


@pytest.fixture()
def siamese_model_path(
    small_model_params, mock_default_config, flow_config, generate_ms2ds_model_flow
):
    path = ASSETS_DIR / "ms2deep_score.hdf5"

    if not path.exists():
        generate_ms2ds_model_flow.run()

    return str(path)


@pytest.fixture()
def siamese_model(siamese_model_path):
    ms2deepscore_model = load_model(siamese_model_path)
    return ms2deepscore_model


@pytest.fixture()
def ms2deepscore_real_predictor(siamese_model):
    ms2deepscore_predictor = MS2DeepScorePredictor(ion_mode="positive", run_id="2")
    ms2deepscore_predictor.model = siamese_model
    return ms2deepscore_predictor


@pytest.fixture()
def tanimoto_scores_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "tanimoto_scores.pkl")


@pytest.fixture()
def tanimoto_scores(tanimoto_scores_path):
    tanimoto_score = pd.read_pickle(tanimoto_scores_path, compression="gzip")
    return tanimoto_score


@pytest.fixture()
def binned_spectra_to_train():
    path = str(ASSETS_DIR / "ms2deepscore" / "to_train" / "binned_spectra.pkl")
    with open(path, "rb") as f:
        binned_spectra = pickle.load(f)
    return binned_spectra


@pytest.fixture
def binned_spectra_to_train_stored(redis_db, binned_spectra_to_train):
    pipe = redis_db.pipeline()
    for spectrum in binned_spectra_to_train:
        pipe.hset(
            f"{BINNED_SPECTRUM_HASHES}_positive",
            spectrum.metadata["spectrum_id"],
            pickle.dumps(spectrum),
        )
    pipe.execute()


@pytest.fixture()
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
def mock_deploy_model_task(monkeypatch):

    import omigami.spectra_matching.ms2deepscore.flows.deploy_model

    monkeypatch.setattr(
        omigami.spectra_matching.ms2deepscore.flows.deploy_model,
        "DeployModel",
        DummyTask,
    )


@pytest.fixture
def generate_ms2ds_model_flow(tmpdir, flow_config, monkeypatch, clean_chunk_files):
    """Used to generate local asset ms2deep_score.hd5 for multiple tests."""
    import omigami.spectra_matching.ms2deepscore.flows.training_flow

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
    spectrum_cleaner = SpectrumCleaner()

    flow_params = TrainingFlowParameters(
        fs_dgw=data_gtw,
        spectrum_dgw=spectrum_dgw,
        source_uri=SOURCE_URI_PARTIAL_GNPS_500_SPECTRA,
        # the three parameters below are for using cached assets instead of downloading
        dataset_directory=str(ASSETS_DIR.parent),
        dataset_name="SMALL_GNPS_500_spectra.json",
        chunk_size=150000,
        ion_mode="positive",
        overwrite_model=True,
        # we use everything but the model path as tmpdir. We only want the model from this script
        scores_output_path=str(tmpdir / "tanimoto_scores.pkl"),
        fingerprint_n_bits=2048,
        scores_decimals=5,
        spectrum_binner_n_bins=10000,
        spectrum_binner_output_path=str(tmpdir / "spectrum_binner.pkl"),
        model_output_path=str(ASSETS_DIR / "ms2deep_score.hdf5"),
        dataset_checkpoint_name="spectrum_ids_500.pkl",
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
