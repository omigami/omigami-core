import feast
import pytest

from spec2vec_mlops import config
from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.load_data import DataLoader
from spec2vec_mlops.tasks.store_cleaned_data import DataStorer


FEAST_CORE_URL = config["feast"]["url"]


@pytest.fixture
def data_storer(tmpdir):
    return DataStorer(str(tmpdir))


@pytest.fixture
def cleaned_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    return dc.clean_data(loaded_data)


def test_create_spectrum_info_table(data_storer):
    client = feast.Client(core_url=FEAST_CORE_URL, telemetry=False)
    data_storer._create_spectrum_info_table(client)
    assert client.list_feature_tables()[0].name == "spectrum_info"


def test_store_cleaned_data(data_storer, cleaned_data):
    pass


