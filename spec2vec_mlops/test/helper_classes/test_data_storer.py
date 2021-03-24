import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import DataStorer
from spec2vec_mlops.tasks.convert_to_documents import DocumentConverter

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skip(
    "These tests can only be run if the Feast docker-compose is up"
)


@pytest.fixture()
def data_storer(tmpdir):
    return DataStorer(f"file://{tmpdir}", FEAST_CORE_URL)


@pytest.fixture
def documents_data(cleaned_data):
    converter = DocumentConverter()
    return [converter.convert_to_document(spectrum, 1) for spectrum in cleaned_data]


def test_create_spectrum_info_table(data_storer):
    data_storer._create_spectrum_info_table()
    assert data_storer.client.list_feature_tables()[0].name == data_storer.feature_table_name


def test_get_cleaned_data_df(data_storer, cleaned_data):
    spectrum_df = data_storer._get_cleaned_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert (
        len(spectrum_df.columns) == len(data_storer.features2types.keys()) + 3
    )  # +3 because of spectrum_id, created_timestamp and event_timestamp


def test_store_cleaned_data(data_storer, cleaned_data):
    data_storer.store_cleaned_data(cleaned_data)


def test_get_words_df(data_storer, documents_data):
    words_df = data_storer._get_words_df(documents_data)
    assert len(words_df) == len(documents_data)
    assert set(words_df.columns) == {"spectrum_id", "words", "event_timestamp"}


def test_store_words(data_storer, documents_data):
    data_storer.store_words(documents_data)
