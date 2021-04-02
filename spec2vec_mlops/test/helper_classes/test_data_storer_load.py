import os

import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import (
    SpectrumIDStorer,
    SpectrumStorer,
    DocumentStorer,
    EmbeddingStorer,
)

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


@pytest.fixture()
def spectrum_ids_storer(tmpdir):
    return SpectrumIDStorer(
        out_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL,
        feature_table_name="spectrum_ids_info",
    )


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


@pytest.fixture()
def target_spectrum_ids(spectrum_ids_storer, cleaned_data):
    ids = [cleaned_data[0].metadata["spectrum_id"]]
    spectrum_ids_storer.store_spectrum_ids(ids)
    return ids


@pytest.fixture()
def spectrum_stored(spectrum_storer, cleaned_data):
    spectrum_storer.store_cleaned_data(cleaned_data)


@pytest.fixture()
def documents_stored(document_storer, documents_data):
    document_storer.store_documents(documents_data)


@pytest.fixture()
def embeddings_stored(embedding_storer, documents_data, embeddings):
    embedding_storer.store_embeddings(documents_data, embeddings, "1")


def test_load_all_spectrum_ids(spectrum_ids_storer, target_spectrum_ids):
    all_spectrum_ids = spectrum_ids_storer.read_spectrum_ids()
    assert len(all_spectrum_ids) == len(target_spectrum_ids)
