import pickle
from pathlib import Path

import boto3
import pytest
import s3fs
from moto import mock_s3

from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.storer_classes import (
    EmbeddingStorer,
    SpectrumIDStorer,
    SpectrumStorer,
    DocumentStorer,
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


@pytest.fixture(scope="module")
def assets_dir():
    return Path(__file__).parents[0] / "assets"


@pytest.fixture(scope="module")
def local_gnps_small_json(assets_dir):
    path = str(assets_dir / "SMALL_GNPS.json")
    return path


@pytest.fixture()
def loaded_data(local_gnps_small_json, tmpdir):
    dl = DataLoader(local_gnps_small_json)
    return dl.load_gnps_json()


@pytest.fixture(scope="module")
def cleaned_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture(scope="module")
def documents_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)
    return documents_data


@pytest.fixture(scope="module")
def word2vec_model(assets_dir):
    path = str(assets_dir / "model.pickle")
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    return model


@pytest.fixture(scope="module")
def embeddings(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_as_embeddings.pickle")
    with open(path, "rb") as handle:
        embeddings = pickle.load(handle)
    return embeddings


@pytest.fixture(scope="module")
def spectrum_ids_storer():
    return SpectrumIDStorer(
        feature_table_name="spectrum_ids_info",
    )


@pytest.fixture(scope="module")
def all_spectrum_ids(spectrum_ids_storer, cleaned_data):
    ids = [spectrum.metadata["spectrum_id"] for spectrum in cleaned_data]
    spectrum_ids_storer.store(ids)
    return ids


@pytest.fixture(scope="module")
def embedding_storer():
    return EmbeddingStorer(
        feature_table_name="embedding_info",
        run_id="1",
    )


@pytest.fixture(scope="module")
def embeddings_stored(embedding_storer, embeddings):
    embedding_storer.store(embeddings)


@pytest.fixture(scope="module")
def spectrum_storer():
    return SpectrumStorer(
        feature_table_name="spectrum_info",
    )


@pytest.fixture(scope="module")
def spectrum_stored(spectrum_storer, cleaned_data):
    spectrum_storer.store(cleaned_data)


@pytest.fixture(scope="module")
def document_storer():
    return DocumentStorer(
        feature_table_name="document_info",
    )


@pytest.fixture(scope="module")
def documents_stored(document_storer, documents_data):
    document_storer.store(documents_data)


@pytest.fixture()
def s3_mock():
    mock = mock_s3()
    mock.start()
    conn = boto3.client("s3")
    conn.create_bucket(
        Bucket="test-bucket",
        CreateBucketConfiguration={"LocationConstraint": "eu-west-1"},
    )
    yield s3fs.S3FileSystem()
    mock.stop()
