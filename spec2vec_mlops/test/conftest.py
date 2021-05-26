import pickle
from pathlib import Path

import boto3
import ijson
import pytest
import s3fs
from moto import mock_s3

from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops import default_configs

TEST_TASK_CONFIG = dict(max_retries=1, retry_delay=0)

KEYS = default_configs["gnps_json"]["necessary_keys"]


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


ASSETS_DIR = Path(__file__).parents[0] / "assets"


@pytest.fixture(scope="module")
def local_gnps_small_json():
    path = str(ASSETS_DIR / "SMALL_GNPS.json")
    return path


@pytest.fixture()
def loaded_data(local_gnps_small_json, tmpdir):
    with open(local_gnps_small_json, "rb") as f:
        items = ijson.items(f, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
    return results


@pytest.fixture(scope="module")
def cleaned_data():
    path = str(ASSETS_DIR / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture(scope="module")
def documents_data():
    path = str(ASSETS_DIR / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)
    return documents_data


@pytest.fixture(scope="module")
def word2vec_model():
    path = str(ASSETS_DIR / "model.pickle")
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    return model


@pytest.fixture(scope="module")
def embeddings():
    path = str(ASSETS_DIR / "SMALL_GNPS_as_embeddings.pickle")
    with open(path, "rb") as handle:
        embeddings = pickle.load(handle)
    return embeddings


@pytest.fixture(scope="module")
def embeddings_2k():
    path = str(ASSETS_DIR / "embeddings_2k.pickle")
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


@pytest.fixture
def spectrum_ids(local_gnps_small_json):
    ids = FSInputDataGateway().get_spectrum_ids(local_gnps_small_json)
    return ids
