import pytest
from feast import ValueType, FeatureTable
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.storer_classes import (
    DocumentStorer,
    EmbeddingStorer,
    Storer,
    SpectrumStorer,
)

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)


pytestmark = pytest.mark.skip("It can only be run if the Feast docker-compose is up")


@pytest.fixture()
def spectrum_storer(tmpdir):
    return SpectrumStorer(
        out_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL,
        feature_table_name="spectrum_info",
    )


@pytest.fixture()
def embedding_storer(tmpdir):
    return EmbeddingStorer(
        out_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL,
        feature_table_name="embedding_info",
    )


@pytest.fixture()
def document_storer(tmpdir):
    return DocumentStorer(
        out_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL,
        feature_table_name="document_info",
    )


def test_storer_get_or_create_table(tmpdir):
    storer = Storer(
        out_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL,
        feature_table_name="test_table_info",
        **{"column1": ValueType.DOUBLE_LIST, "column2": ValueType.STRING},
    )
    table = storer.get_or_create_table("some_id", "Test_table identifier")
    existing_tables = [table.name for table in storer.client.list_feature_tables()]
    assert isinstance(table, FeatureTable)
    assert storer.feature_table_name in existing_tables


def test_spectrum_storer_get_cleaned_data_df(spectrum_storer, cleaned_data):
    spectrum_df = spectrum_storer._get_cleaned_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert (
        len(spectrum_df.columns) == len(spectrum_storer.features2types.keys()) + 3
    )  # +3 because of spectrum_id, created_timestamp and event_timestamp
    assert not spectrum_df.spectrum_id.isnull().any()
    assert not spectrum_df.event_timestamp.isnull().any()
    assert not spectrum_df.created_timestamp.isnull().any()
    assert is_datetime(spectrum_df.event_timestamp)
    assert is_datetime(spectrum_df.created_timestamp)


def test_spectrum_storer_store_cleaned_data(spectrum_storer, cleaned_data):
    spectrum_storer.store_cleaned_data(cleaned_data)


def test_document_storer_get_documents_df(document_storer, documents_data):
    documents_df = document_storer._get_documents_df(documents_data)
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


def test_document_storer_store_documents(document_storer, documents_data):
    document_storer.store_documents(documents_data)


def test_embedding_storer_get_embedding_df(
    embedding_storer, documents_data, embeddings
):
    embedding_df = embedding_storer._get_embedding_df(embeddings, "run_id")
    assert len(embedding_df) == len(documents_data)
    assert set(embedding_df.columns) == {
        "spectrum_id",
        "embedding",
        "run_id",
        "event_timestamp",
        "create_timestamp",
    }
    assert not embedding_df.spectrum_id.isnull().any()
    assert not embedding_df.embedding.isnull().any()
    assert not embedding_df.event_timestamp.isnull().any()


def test_embedding_storer_store_embeddings(
    embedding_storer, documents_data, embeddings
):
    embedding_storer.store_embeddings(embeddings, "1")