import pickle
import shutil
from pathlib import Path
from time import sleep

import boto3
import ijson
import mlflow
import pandas as pd
import pytest
import s3fs
from moto import mock_s3
from prefect import Client, Flow
from pytest_redis import factories

import omigami
import omigami.utils
from omigami.authentication.prefect_factory import prefect_client_factory
from omigami.config import (
    MLFLOW_SERVER,
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    EMBEDDING_HASHES,
)
from omigami.flow_config import make_flow_config, PrefectExecutorMethods
from omigami.spectra_matching.ms2deepscore.config import BINNED_SPECTRUM_HASHES
from omigami.spectra_matching.ms2deepscore.embedding import MS2DeepScoreEmbedding
from omigami.spectra_matching.storage import FSDataGateway, KEYS
from omigami.spectra_matching.tasks import (
    ChunkingParameters,
    CreateChunks,
    CleanRawSpectraParameters,
    CleanRawSpectra,
)

ASSETS_DIR = Path(__file__).parents[1] / "assets"
TEST_TASK_CONFIG = dict(max_retries=1, retry_delay=pd.Timedelta(seconds=0.1))
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
def single_spectrum_as_json(local_gnps_small_json):
    with open(local_gnps_small_json, "rb") as f:
        items = ijson.items(f, "item", multiple_values=True)
        results = [{k: item[k] for k in KEYS} for item in items]
    return results


@pytest.fixture(scope="module")
def spectrum_ids_by_mode(single_spectrum_as_json):
    spectrum_ids = {"positive": [], "negative": []}

    for spectrum in single_spectrum_as_json:
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


@pytest.fixture(scope="module")
def cleaned_data_ms2deep_score():
    path = str(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture(scope="module")
def binned_spectra(cleaned_data_ms2deep_score):
    """
    This fixture creates ms2deepscore.BinnedSpectrum objects from cleaned_data_ms2deep_score.
    If the pkl file is already available it load binned_spectra from it.

    Parameters
    ----------
    cleaned_data_ms2deep_score: List[Spectrum]

    Returns
    -------
    List[BinnedSpectrum]

    """
    from ms2deepscore import SpectrumBinner

    path = Path(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_as_binned_spectra.pkl")
    if path.exists():
        with open(str(path), "rb") as f:
            binned_spectra = pickle.load(f)
    else:
        spectrum_binner = SpectrumBinner(number_of_bins=10000)
        binned_spectra = spectrum_binner.fit_transform(cleaned_data_ms2deep_score)
        for binned_spectrum, spectrum in zip(
            binned_spectra, cleaned_data_ms2deep_score
        ):
            binned_spectrum.set("spectrum_id", spectrum.get("spectrum_id"))
            binned_spectrum.set("inchi", spectrum.get("inchi"))
        with open(str(path), "wb") as f:
            pickle.dump(binned_spectra, f)
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
    path = str(ASSETS_DIR / "ms2deepscore" / "SMALL_GNPS_as_embeddings.pkl")

    with open(path, "rb") as handle:
        data = pickle.load(handle)

    embeddings = [MS2DeepScoreEmbedding(**d) for d in data]
    return embeddings


@pytest.fixture()
def ms2deepscore_embeddings_stored(redis_db, embeddings_from_real_predictor):
    project = "ms2deepscore"
    ion_mode = "positive"
    pipe = redis_db.pipeline()
    for embedding in embeddings_from_real_predictor:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{project}_{ion_mode}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


@pytest.fixture()
def redis_full_setup(
    spectra_stored,
    binned_spectra_stored,
    ms2deepscore_embeddings_stored,
):
    pass


@pytest.fixture()
def s3_documents_directory():
    return "s3://test-bucket/documents"


@pytest.fixture()
def clean_chunk_files():
    chunk_root = ASSETS_DIR / "raw"
    if chunk_root.exists():
        shutil.rmtree(chunk_root)

    cleaned_root = ASSETS_DIR / "cleaned"
    if cleaned_root.exists():
        shutil.rmtree(cleaned_root)


@pytest.fixture()
def fitted_spectrum_binner_path():
    return str(ASSETS_DIR / "ms2deepscore" / "to_train" / "fitted_spectrum_binner.pkl")


@pytest.fixture()
def fitted_spectrum_binner(fitted_spectrum_binner_path):
    with open(fitted_spectrum_binner_path, "rb") as f:
        spectrum_binner = pickle.load(f)
    return spectrum_binner


@pytest.fixture()
def backend_services():
    """Connects to mlflow and prefect and return client objects to communicate with them"""
    mlflow.set_tracking_uri(MLFLOW_SERVER)
    mlflow_client = mlflow.tracking.MlflowClient()

    prefect_client = prefect_client_factory.get()

    if not prefect_client.active_tenant_id:
        prefect_client.create_tenant("default")

    return {"prefect": prefect_client, "mlflow": mlflow_client}


@pytest.fixture
def flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-hermione-XXII",
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="0",
    )
    return flow_config


def monitor_flow_results(client: Client, flow_run_id: str):
    flow_duration = 0
    while not (
        client.get_flow_run_state(flow_run_id).is_successful()
        or client.get_flow_run_state(flow_run_id).is_failed()
    ):
        flow_duration += 0.5
        sleep(0.5)

        if flow_duration > 60:
            raise TimeoutError("Flow timeout. Check flow logs.")


@pytest.fixture
def create_chunks_task(clean_chunk_files, local_gnps_small_json):
    data_gtw = FSDataGateway()
    output_directory = ASSETS_DIR / "raw" / "positive"
    chunking_parameters = ChunkingParameters(
        local_gnps_small_json, str(output_directory), 150000, "positive"
    )
    t = CreateChunks(data_gtw=data_gtw, chunking_parameters=chunking_parameters)

    return t


@pytest.fixture
def clean_spectra_task():
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)
    t = CleanRawSpectra(fs_dgw, params)

    return t


@pytest.fixture()
def cleaned_spectra_paths(create_chunks_task, clean_spectra_task):
    with Flow("test-flow") as flow:
        cc = create_chunks_task()
        cs = clean_spectra_task.map(cc)

    res = flow.run()

    return res.result[cs].result


@pytest.fixture
def cleaned_spectra_chunks(cleaned_spectra_paths):
    fs_dgw = FSDataGateway()
    return [fs_dgw.read_from_file(p) for p in cleaned_spectra_paths]
