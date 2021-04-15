import pickle
from pathlib import Path

import pytest

from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.storer_classes import (
    EmbeddingStorer,
    SpectrumIDStorer,
)


def pytest_addoption(parser):
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable longrun-decorated tests",
    )


def pytest_configure(config):
    if not config.option.longrun:
        setattr(config.option, "markexpr", "not longrun")


@pytest.fixture()
def assets_dir():
    return Path(__file__).parents[0] / "assets"


@pytest.fixture()
def gnps_small_json(assets_dir):
    path = str(assets_dir / "SMALL_GNPS.json")
    return f"file://{path}"


@pytest.fixture()
def local_gnps_small_json(assets_dir):
    path = str(assets_dir / "SMALL_GNPS.json")
    return path


@pytest.fixture()
def loaded_data(local_gnps_small_json, tmpdir):
    dl = DataLoader(local_gnps_small_json)
    return dl.load_gnps_json()


@pytest.fixture()
def cleaned_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture()
def documents_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)
    return documents_data


@pytest.fixture()
def word2vec_model(assets_dir):
    path = str(assets_dir / "model.pickle")
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    return model


@pytest.fixture()
def embeddings(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_as_embeddings.pickle")
    with open(path, "rb") as handle:
        embeddings = pickle.load(handle)
    return embeddings


@pytest.fixture()
def spectrum_ids_storer(tmpdir):
    return SpectrumIDStorer(
        feature_table_name="spectrum_ids_info",
    )


@pytest.fixture()
def all_spectrum_ids(spectrum_ids_storer, cleaned_data):
    ids = [spectrum.metadata["spectrum_id"] for spectrum in cleaned_data]
    spectrum_ids_storer.store(ids)
    return ids


@pytest.fixture()
def embedding_storer(tmpdir):
    return EmbeddingStorer(
        feature_table_name="embedding_info",
        run_id="1",
    )


@pytest.fixture()
def embeddings_stored(embedding_storer, embeddings):
    embedding_storer.store(embeddings)
