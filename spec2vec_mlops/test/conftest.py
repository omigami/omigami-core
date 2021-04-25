import pickle
from pathlib import Path

import boto3
import pytest
import s3fs
from moto import mock_s3

from spec2vec_mlops.helper_classes.data_loader import DataLoader


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
