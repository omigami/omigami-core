import pickle
from pathlib import Path

import boto3
import ijson
import pytest
import s3fs
from drfs.filesystems import get_fs
from moto import mock_s3
from pytest_redis import factories

import omigami
import omigami.config
from omigami.spec2vec.config import (
    DOCUMENT_HASHES,
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    EMBEDDING_HASHES,
)
from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway, KEYS

ASSETS_DIR = Path(__file__).parents[0] / "assets"
TEST_TASK_CONFIG = dict(max_retries=1, retry_delay=0)
redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture
def mock_default_config(monkeypatch):
    monkeypatch.setattr(omigami.config, "DEFAULT_PREFECT_TASK_CONFIG", TEST_TASK_CONFIG)


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
def local_gnps_small_json():
    path = str(ASSETS_DIR / "SMALL_GNPS.json")
    return path


@pytest.fixture(scope="module")
def loaded_data(local_gnps_small_json):
    with open(local_gnps_small_json, "rb") as f:
        items = ijson.items(f, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
    return results


@pytest.fixture(scope="module")
def spectrum_ids_by_mode(loaded_data):
    spectrum_ids = {"positive": [], "negative": []}

    for spectrum in loaded_data:
        spectrum_ids[spectrum["Ion_Mode"].lower()].append(spectrum["spectrum_id"])
    return spectrum_ids


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
    path = str(ASSETS_DIR / "word2vec_model.pickle")
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


@pytest.fixture
def spectra_stored(redis_db, cleaned_data):
    pipe = redis_db.pipeline()
    for spectrum in cleaned_data:
        pipe.zadd(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
            {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
        )
        pipe.hset(
            SPECTRUM_HASHES, spectrum.metadata["spectrum_id"], pickle.dumps(spectrum)
        )
    pipe.execute()


@pytest.fixture()
def documents_stored(redis_db, cleaned_data, documents_data):
    pipe = redis_db.pipeline()
    for i, document in enumerate(documents_data):
        pipe.hset(
            DOCUMENT_HASHES,
            cleaned_data[i].metadata["spectrum_id"],
            pickle.dumps(document),
        )
    pipe.execute()


@pytest.fixture()
def embeddings_stored(redis_db, cleaned_data, embeddings):
    run_id = "1"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


@pytest.fixture()
def redis_full_setup(spectra_stored, documents_stored, embeddings_stored):
    pass


@pytest.fixture()
def clean_chunk_files():
    fs = get_fs(str(ASSETS_DIR))
    _ = [fs.rm(f) for f in fs.ls(ASSETS_DIR / "chunks" / "positive")]
    _ = [fs.rm(f) for f in fs.ls(ASSETS_DIR / "chunks" / "negative")]
