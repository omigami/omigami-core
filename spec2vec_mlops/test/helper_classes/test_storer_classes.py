import os

import pytest
from feast import ValueType, FeatureTable
from pandas.api.types import is_datetime64_any_dtype as is_datetime

from spec2vec_mlops.helper_classes.feast_dgw import FeastDataGateway

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_FEAST_TEST", True),
    reason="It can only be run if the Feast docker-compose is up",
)


def test_feast_table_get_or_create_table(tmpdir):
    dgw = FeastDataGateway()
    table = dgw.get_or_create_table(
        "Test_table identifier",
        "some_id",
        "test_table_info",
        {"column1": ValueType.DOUBLE_LIST, "column2": ValueType.STRING},
    )
    existing_tables = [table.name for table in dgw.client.list_feature_tables()]
    assert isinstance(table, FeatureTable)
    assert table.name in existing_tables


def test_spectrum_ids_storer_store_spectrum_ids(spectrum_ids_storer, cleaned_data):
    spectrum_ids = [spectrum.metadata["spectrum_id"] for spectrum in cleaned_data]
    spectrum_ids_storer.store(spectrum_ids)


def test_spectrum_storer_get_data_df(spectrum_storer, cleaned_data):
    spectrum_df = spectrum_storer._get_data_df(cleaned_data)
    assert len(spectrum_df) == len(cleaned_data)
    assert not spectrum_df.spectrum_id.isnull().any()
    assert not spectrum_df.event_timestamp.isnull().any()
    assert not spectrum_df.created_timestamp.isnull().any()
    assert is_datetime(spectrum_df.event_timestamp)
    assert is_datetime(spectrum_df.created_timestamp)


def test_spectrum_storer_store_cleaned_data(spectrum_storer, cleaned_data):
    spectrum_storer.store(cleaned_data)


def test_document_storer_get_data_df(document_storer, documents_data):
    documents_df = document_storer._get_data_df(documents_data)
    assert set(documents_df.columns) == {
        "spectrum_id",
        "words",
        "losses",
        "weights",
        "event_timestamp",
        "created_timestamp",
        "n_decimals",
    }
    assert not documents_df.spectrum_id.isnull().any()
    assert not documents_df.words.isnull().any()
    assert not documents_df.event_timestamp.isnull().any()


def test_document_storer_store_documents(document_storer, documents_data):
    document_storer.store(documents_data)


def test_embedding_storer_get_data_df(embedding_storer, documents_data, embeddings):
    embedding_df = embedding_storer._get_data_df(embeddings)
    assert len(embedding_df) == len(documents_data)
    assert set(embedding_df.columns) == {
        "spectrum_id",
        "embedding",
        "run_id",
        "event_timestamp",
        "created_timestamp",
    }
    assert not embedding_df.spectrum_id.isnull().any()
    assert not embedding_df.embedding.isnull().any()
    assert not embedding_df.event_timestamp.isnull().any()


def test_embedding_storer_store_embeddings(
    embedding_storer, documents_data, embeddings
):
    embedding_storer.store(embeddings)
