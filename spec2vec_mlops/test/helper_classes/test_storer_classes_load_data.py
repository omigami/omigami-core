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


@pytest.mark.slow
def test_load_all_spectrum_ids_offline(spectrum_ids_storer, target_spectrum_ids):
    all_spectrum_ids = spectrum_ids_storer.read_offline()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)

    # ingest it again and assert that there is no duplicate
    target_id = target_spectrum_ids[0]
    spectrum_ids_storer.store([target_id])
    updated_spectrum_ids = spectrum_ids_storer.read_offline()
    assert list(updated_spectrum_ids).count(target_id) == 1


@pytest.mark.slow
def test_load_spectrum_offline(spectrum_storer, spectrum_stored, target_spectrum_ids):
    spectra = spectrum_storer.read_offline(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)


@pytest.mark.slow
def test_load_documents_offline(document_storer, documents_stored, target_spectrum_ids):
    documents = document_storer.read_offline(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)


@pytest.mark.slow
def test_load_embeddings_offline(
    embedding_storer, embeddings_stored, target_spectrum_ids
):
    embeddings = embedding_storer.read_offline(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)


def test_load_all_spectrum_ids_online(spectrum_ids_storer, target_spectrum_ids):
    spectrum_ids_storer.store_online()
    all_spectrum_ids = spectrum_ids_storer.read_online()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)


def test_load_spectrum_online(spectrum_storer, spectrum_stored, target_spectrum_ids):
    spectrum_storer.store_online()
    spectra = spectrum_storer.read_online(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)


def test_load_documents_online(document_storer, documents_stored, target_spectrum_ids):
    document_storer.store_online()
    documents = document_storer.read_online(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)


def test_load_embeddings_online(
    embedding_storer, embeddings_stored, target_spectrum_ids
):
    embedding_storer.store_online()
    embeddings = embedding_storer.read_online(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)


@pytest.mark.skip(reason="run this test only after online store is fully loaded.")
def test_load_online(
    spectrum_ids_storer, spectrum_storer, document_storer, embedding_storer
):
    target_spectrum_ids = ["CCMSLIB00000001547"]
    all_spectrum_ids = spectrum_ids_storer.read_online()
    assert all(id in all_spectrum_ids for id in target_spectrum_ids)

    spectra = spectrum_storer.read_online(target_spectrum_ids)
    assert len(spectra) == len(target_spectrum_ids)
    assert len(spectra[0].peaks.mz) > 0

    documents = document_storer.read_online(target_spectrum_ids)
    assert len(documents) == len(target_spectrum_ids)
    assert documents[0].n_decimals

    embeddings = embedding_storer.read_online(target_spectrum_ids)
    assert len(embeddings) == len(target_spectrum_ids)
    assert embeddings[0].n_decimals
