import os

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


@pytest.fixture(scope="module")
def target_spectrum_ids(spectrum_ids_storer, cleaned_data):
    ids = [cleaned_data[0].metadata["spectrum_id"]]
    spectrum_ids_storer.store(ids)
    return ids


def test_load_all_spectrum_ids_offline(spectrum_ids_storer, target_spectrum_ids):
    all_spectrum_ids = spectrum_ids_storer.read_offline()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)

    # ingest it again and assert that there is no duplicate
    target_id = target_spectrum_ids[0]
    spectrum_ids_storer.store([target_id])
    updated_spectrum_ids = spectrum_ids_storer.read_offline()
    assert list(updated_spectrum_ids).count(target_id) == 1


def test_load_spectrum_offline(spectrum_storer, spectrum_stored, target_spectrum_ids):
    spectra = spectrum_storer.read_offline(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)


def test_load_documents_offline(document_storer, documents_stored, target_spectrum_ids):
    documents = document_storer.read_offline(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)


def test_load_embeddings_offline(
    embedding_storer, embeddings_stored, target_spectrum_ids
):
    embeddings = embedding_storer.read_offline(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)


def test_load_all_spectrum_ids_online(spectrum_ids_storer, target_spectrum_ids):
    all_spectrum_ids = spectrum_ids_storer.read_online()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)


def test_load_spectrum_online(spectrum_storer, spectrum_stored, target_spectrum_ids):
    spectra = spectrum_storer.read_online(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)


def test_load_documents_online(document_storer, documents_stored, target_spectrum_ids):
    documents = document_storer.read_online(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)


def test_load_embeddings_online(
    embedding_storer, embeddings_stored, target_spectrum_ids
):
    embeddings = embedding_storer.read_online(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)
