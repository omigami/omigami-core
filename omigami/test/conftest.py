import itertools
import os
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
import omigami.utils
from omigami.base.gateways.fs_data_gateway import FSDataGateway, KEYS
from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.spec2vec.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    EMBEDDING_HASHES,
    PROJECT_NAME,
)
from omigami.spec2vec.gateways.redis_spectrum_document import (
    RedisSpectrumDocumentDataGateway,
)

ASSETS_DIR = Path(__file__).parents[0] / "assets"
TEST_TASK_CONFIG = dict(max_retries=1, retry_delay=0)
redis_db = factories.redisdb("redis_nooproc")


@pytest.fixture
def mock_default_config(monkeypatch):
    monkeypatch.setattr(omigami.utils, "DEFAULT_PREFECT_TASK_CONFIG", TEST_TASK_CONFIG)


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
def local_gnps_small_json() -> str:
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
def common_cleaned_data():
    path = str(ASSETS_DIR / "SMALL_GNPS_common_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture(scope="module")
def cleaned_data():
    path = str(ASSETS_DIR / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


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
    ids = FSDataGateway().get_spectrum_ids(local_gnps_small_json)
    return ids


@pytest.fixture
def spectra_stored(redis_db, common_cleaned_data):
    pipe = redis_db.pipeline()
    for spectrum in common_cleaned_data:
        pipe.zadd(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
            {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
        )
        pipe.hset(
            SPECTRUM_HASHES, spectrum.metadata["spectrum_id"], pickle.dumps(spectrum)
        )
    pipe.execute()


@pytest.fixture()
def embeddings_stored(redis_db, cleaned_data, embeddings):
    run_id = "1"
    project = "spec2vec"
    ion_mode = "positive"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{project}_{ion_mode}_{run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


@pytest.fixture(scope="module")
def binned_spectra():
    path = str(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_as_binned_spectra.pickle")
    with open(path, "rb") as handle:
        binned_spectra = pickle.load(handle)
    return binned_spectra


@pytest.fixture
def binned_spectra_stored(redis_db, binned_spectra):
    pipe = redis_db.pipeline()
    for spectrum in binned_spectra:
        pipe.hset(
            f"{BINNED_SPECTRUM_HASHES}_positive",
            spectrum.metadata["spectrum_id"],
            pickle.dumps(spectrum),
        )
    pipe.execute()


@pytest.fixture()
def embeddings_from_real_predictor():
    path = str(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_as_embeddings.pickle")
    with open(path, "rb") as handle:
        embeddings = pickle.load(handle)
    return embeddings


@pytest.fixture()
def ms2deepscore_embeddings_stored(redis_db, embeddings_from_real_predictor):
    run_id = "2"
    project = "ms2deepscore"
    ion_mode = "positive"
    pipe = redis_db.pipeline()
    for embedding in embeddings_from_real_predictor:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{project}_{ion_mode}_{run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


@pytest.fixture()
def redis_full_setup(
    spectra_stored,
    documents_stored,
    embeddings_stored,
    binned_spectra_stored,
    ms2deepscore_embeddings_stored,
):
    pass


@pytest.fixture(scope="module")
def documents_data():
    path = str(ASSETS_DIR / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)

    return documents_data


@pytest.fixture()
def documents_stored(s3_documents_directory, documents_data, s3_mock):
    chunk_size = 10

    documents_data = [
        documents_data[i : i + chunk_size]
        for i in range(0, len(documents_data), chunk_size)
    ]

    dgw = RedisSpectrumDocumentDataGateway(project=PROJECT_NAME)
    fs_dgw = FSDataGateway()

    fs = get_fs(s3_documents_directory)
    if not os.path.exists(s3_documents_directory):
        fs.makedirs(s3_documents_directory)

    for i, documents in enumerate(documents_data):
        doc_path = f"{s3_documents_directory}/test{i}.pickle"
        dgw.write_documents(documents, "positive")
        fs_dgw.serialize_to_file(doc_path, documents)

    return list(itertools.chain.from_iterable(documents_data))


@pytest.fixture()
def s3_documents_directory():
    return "s3://test-bucket/documents"


@pytest.fixture()
def clean_chunk_files():
    fs = get_fs(str(ASSETS_DIR))
    _ = [fs.rm(f) for f in fs.ls(ASSETS_DIR / "chunks" / "positive")]
    _ = [fs.rm(f) for f in fs.ls(ASSETS_DIR / "chunks" / "negative")]


@pytest.fixture()
def fitted_spectrum_binner_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "fitted_spectrum_binner.pkl")


@pytest.fixture()
def fitted_spectrum_binner(fitted_spectrum_binner_path):
    with open(fitted_spectrum_binner_path, "rb") as f:
        spectrum_binner = pickle.load(f)
    return spectrum_binner
