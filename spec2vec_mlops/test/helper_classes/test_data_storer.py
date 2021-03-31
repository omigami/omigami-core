import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import DataStorer

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)


@pytest.fixture()
def data_storer(tmpdir):
    return DataStorer(f"file://{tmpdir}", FEAST_CORE_URL)


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_create_spectrum_info_table(data_storer):
    data_storer._create_spectrum_info_table()
    assert (
        data_storer.client.list_feature_tables()[0].name
        == data_storer.feature_table_name
    )


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_get_cleaned_data_df(data_storer, cleaned_data):
    spectrum_df = data_storer._get_cleaned_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert (
        len(spectrum_df.columns) == len(data_storer.features2types.keys()) + 3
    )  # +3 because of spectrum_id, created_timestamp and event_timestamp
    assert not spectrum_df.spectrum_id.isnull().any()
    assert not spectrum_df.event_timestamp.isnull().any()
    assert not spectrum_df.created_timestamp.isnull().any()


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_store_cleaned_data(data_storer, cleaned_data):
    data_storer.store_cleaned_data(cleaned_data)


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_get_documents_df(data_storer, documents_data):
    documents_df = data_storer._get_documents_df(documents_data)
    assert set(documents_df.columns) == {
        "spectrum_id",
        "words",
        "losses",
        "weights",
        "event_timestamp",
    }
    assert not documents_df.spectrum_id.isnull().any()
    assert not documents_df.words.isnull().any()
    assert not documents_df.event_timestamp.isnull().any()


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_store_documents(data_storer, documents_data):
    data_storer.store_documents(documents_data)


def test_get_embeddings_df(data_storer, documents_data, embeddings):
    embeddings_df = data_storer._get_embeddings_df(documents_data, embeddings)
    assert set(embeddings_df.columns) == {
        "spectrum_id",
        "embeddings",
        "event_timestamp",
    }
    assert not embeddings_df.spectrum_id.isnull().any()
    assert not embeddings_df.embeddings.isnull().any()
    assert not embeddings_df.event_timestamp.isnull().any()


@pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_store_embeddings(data_storer, documents_data, embeddings):
    data_storer.store_embeddings(documents_data, embeddings)

