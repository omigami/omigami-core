import pytest
from feast import Client

from spec2vec_mlops import config
from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.load_data import DataLoader
from spec2vec_mlops.tasks.store_cleaned_data import DataStorer


FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skip(
    "These tests can only be run if the Feast docker-compose is up"
)


@pytest.fixture
def data_storer(tmpdir):
    return DataStorer(f"file://{tmpdir}", FEAST_CORE_URL)


@pytest.fixture
def cleaned_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    return dc.clean_data(loaded_data)


def test_create_spectrum_info_table(data_storer):
    client = Client(core_url=FEAST_CORE_URL, telemetry=False)
    data_storer._create_spectrum_info_table(client)
    assert client.list_feature_tables()[0].name == data_storer.feature_table_name


def test_get_data_df(data_storer, cleaned_data):
    spectrum_df = data_storer._get_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert (
        len(spectrum_df.columns) == len(data_storer.features2types.keys()) + 3
    )  # +3 because of spectrum_id, created_timestamp and event_timestamp


def test_store_cleaned_data(data_storer, cleaned_data):
    data_storer.store_cleaned_data(cleaned_data)
