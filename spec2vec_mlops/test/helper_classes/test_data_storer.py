import os

import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import DataStorer

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_FEAST_TEST", True),
    reason="It can only be run if the Feast docker-compose is up",
)


@pytest.fixture()
def data_storer(tmpdir):
    return DataStorer(f"file://{tmpdir}", FEAST_CORE_URL)


def test_create_spectrum_entity(data_storer):
    data_storer._create_spectrum_entity()
    assert data_storer.feature_entity_name in [
        e.name for e in data_storer.client.list_entities()
    ]


def test_create_spectrum_meta_entity(data_storer):
    data_storer._create_meta_entity()
    assert data_storer.meta_entity_name in [
        e.name for e in data_storer.client.list_entities()
    ]


def test_create_spectrum_table(data_storer):
    data_storer._create_spectrum_info_table()
    assert (
        data_storer.client.get_feature_table(data_storer.feature_table_name).name
        == data_storer.feature_table_name
    )


def test_create_meta_table(data_storer):
    data_storer._create_spectrum_meta_table()
    assert (
        data_storer.client.get_feature_table(data_storer.meta_table_name).name
        == data_storer.meta_table_name
    )


def test_get_cleaned_data_df(data_storer, cleaned_data):
    spectrum_df = data_storer._get_cleaned_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert (
        len(spectrum_df.columns) == len(data_storer.features2types.keys()) + 3
    )  # +3 because of spectrum_id, created_timestamp and event_timestamp


def test_store_cleaned_data(data_storer, cleaned_data):
    data_storer.store_cleaned_data(cleaned_data)


def test_get_words_df(data_storer, documents_data):
    documents_df = data_storer._get_documents_df(documents_data)
    assert set(documents_df.columns) == {
        "spectrum_id",
        "words",
        "losses",
        "weights",
        "event_timestamp",
        "created_timestamp",
    }
    assert not documents_df.spectrum_id.isnull().any()
    assert not documents_df.words.isnull().any()


def test_store_documents(data_storer, documents_data):
    data_storer.store_documents(documents_data)
