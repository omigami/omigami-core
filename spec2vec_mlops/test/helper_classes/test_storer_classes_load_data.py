import os

import pytest
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    SpectrumStorer,
    DocumentStorer,
    EmbeddingStorer,
)

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


@pytest.fixture()
def spectrum_ids_storer(tmpdir):
    return SpectrumIDStorer(
        feature_table_name="spectrum_ids_info",
    )


@pytest.fixture()
def spectrum_storer(tmpdir):
    return SpectrumStorer(
        feature_table_name="spectrum_info",
    )


@pytest.fixture()
def embedding_storer(tmpdir):
    return EmbeddingStorer(
        feature_table_name="embedding_info",
        run_id="1",
    )


@pytest.fixture()
def document_storer(tmpdir):
    return DocumentStorer(
        feature_table_name="document_info",
    )


@pytest.fixture()
def target_spectrum_ids(spectrum_ids_storer, cleaned_data):
    ids = [cleaned_data[0].metadata["spectrum_id"]]
    spectrum_ids_storer.store(ids)
    return ids


@pytest.fixture()
def spectrum_stored(spectrum_storer, cleaned_data):
    spectrum_storer.store(cleaned_data)


@pytest.fixture()
def documents_stored(document_storer, documents_data):
    document_storer.store(documents_data)


@pytest.fixture()
def embeddings_stored(embedding_storer, embeddings):
    embedding_storer.store(embeddings)


def test_load_all_spectrum_ids(spectrum_ids_storer, target_spectrum_ids):
    all_spectrum_ids = spectrum_ids_storer.read()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)


def test_load_spectrum(spectrum_storer, spectrum_stored, target_spectrum_ids):
    spectra = spectrum_storer.read(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)


def test_load_documents(document_storer, documents_stored, target_spectrum_ids):
    documents = document_storer.read(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)


def test_load_embeddings(embedding_storer, embeddings_stored, target_spectrum_ids):
    embeddings = embedding_storer.read(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)
