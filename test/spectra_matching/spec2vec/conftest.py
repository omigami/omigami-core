import itertools
import os
import pickle
from pathlib import Path

import pytest
from drfs.filesystems import get_fs
from spec2vec import calc_vector
from spec2vec.model_building import train_new_word2vec_model

from omigami.config import EMBEDDING_HASHES, MLFLOW_DIRECTORY
from omigami.spectra_matching.spec2vec.config import PREDICTOR_ENV_PATH
from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway
from test.spectra_matching.conftest import ASSETS_DIR
from test.spectra_matching.tasks import DummyTask


@pytest.fixture()
def documents_stored(s3_documents_directory, documents_data, s3_mock):
    chunk_size = 10

    documents_data = [
        documents_data[i : i + chunk_size]
        for i in range(0, len(documents_data), chunk_size)
    ]

    fs_dgw = FSDataGateway()

    fs = get_fs(s3_documents_directory)
    if not os.path.exists(s3_documents_directory):
        fs.makedirs(s3_documents_directory)

    for i, documents in enumerate(documents_data):
        doc_path = f"{s3_documents_directory}/test{i}.pickle"
        fs_dgw.serialize_to_file(doc_path, documents)

    return list(itertools.chain.from_iterable(documents_data))


@pytest.fixture()
def spec2vec_embeddings_stored(redis_db, cleaned_data, spec2vec_embeddings):
    project = "spec2vec"
    ion_mode = "positive"
    pipe = redis_db.pipeline()
    for embedding in spec2vec_embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{project}_{ion_mode}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


@pytest.fixture(scope="module")
def spec2vec_embeddings(documents_data, word2vec_model):
    """
    This fixture returns Spec2VecEmbedding objects either by loading from ASSETS_DIR if
    it exists already, or, takes both `documents_data` (where `n_decimals` equals 1) and
    `word2vec_model` to create the Spec2VecEmbedding objects out of them.

    If you do rename Spec2VecEmbedding from omigami/spec2vec/entities/embedding.py,
    please delete the current pkl file and run a test that uses this fixture.
    Then the pkl file will be updated with the new class name.

    Parameters
    ----------
    documents_data: List[SpectrumDocument]
    word2vec_model: Word2Vec model

    Returns
    -------
    List[Spec2VecEmbedding]

    """
    embeddings_path = Path(ASSETS_DIR / "SMALL_GNPS_as_embeddings.pkl")
    if embeddings_path.exists():
        with open(embeddings_path, "rb") as handle:
            embeddings = pickle.load(handle)
    else:
        embeddings = []
        for document in documents_data:
            vector = calc_vector(
                model=word2vec_model,
                document=document,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=15.0,
            )
            embeddings.append(
                Spec2VecEmbedding(
                    vector=vector,
                    spectrum_id=document.get("spectrum_id"),
                    n_decimals=1,
                )
            )
        with open(embeddings_path, "wb") as f:
            pickle.dump(embeddings, f)
    return embeddings


@pytest.fixture(scope="module")
def word2vec_model(documents_data):
    """
    This fixture will only work with `SpectrumDocument`s where `n_decimals` equal to 1,
    because it is trained on this data. If model doesn't exist then it will be trained
    and saved to ASSET_DIR. If exists it will be read from ASSET_DIR.

    Parameters
    ----------
    documents_data: List of `SpectrumDocument` with `n_decimals` set to 1.

    Returns
    -------
    Word2Vec model

    """
    model_path = Path(ASSETS_DIR / "word2vec_model.pkl")
    if model_path.exists():
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = train_new_word2vec_model(documents_data, size=300, iterations=3)
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
    return model


@pytest.fixture()
def spec2vec_redis_setup(
    spectra_stored,
    documents_stored,
    spec2vec_embeddings_stored,
):
    pass


@pytest.fixture(scope="module")
def documents_data():
    """
    Fixture containing `SpectrumDocument` objects whose `n_decimals` is 1.

    Returns
    -------
    List of `SpectrumDocument`s

    """
    path = str(ASSETS_DIR / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)

    return documents_data


@pytest.fixture
def mock_s2v_deploy_model_task(monkeypatch):

    import omigami.spectra_matching.spec2vec.flows.deploy_model
    import omigami.spectra_matching.spec2vec.flows.training_flow

    class DeployModel(DummyTask):
        pass

    monkeypatch.setattr(
        omigami.spectra_matching.spec2vec.flows.deploy_model,
        "DeployModel",
        DeployModel,
    )


@pytest.fixture
def registered_s2v_model(word2vec_model):
    dgw = MLFlowDataGateway()
    model = Spec2VecPredictor(
        model=word2vec_model,
        ion_mode="positive",
        n_decimals=1,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=15,
    )
    model_run_id = dgw.register_model(
        model=model,
        run_name="run",
        experiment_name="local-integration-test-s2v",
        model_name="integration-test-model",
        conda_env_path=PREDICTOR_ENV_PATH,
        experiment_path=str(MLFLOW_DIRECTORY),
    )

    return {"run_id": model_run_id, "model": model}
