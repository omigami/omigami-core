from spec2vec import SpectrumDocument

from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
import pytest
import os


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_document_ids(saved_documents):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()

    stored_ids = redis_dgw.list_document_ids(ion_mode)

    assert len(saved_documents) == len(stored_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_document_ids(cleaned_data):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    spectrum_document_data = [
        SpectrumDocument(spectrum, 2) for spectrum in cleaned_data
    ]

    redis_dgw.write_document_ids(spectrum_document_data, ion_mode)

    stored_ids = list(redis_dgw.list_document_ids(ion_mode))
    ids = [doc.get("spectrum_id") for doc in spectrum_document_data]
    stored_ids.sort()
    ids.sort()
    assert stored_ids == ids


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_remove_document_ids(saved_documents):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()

    document_ids = [doc.get("spectrum_id") for doc in saved_documents]

    redis_dgw.remove_document_ids(document_ids[:50], ion_mode)

    leftover_ids = list(redis_dgw.list_document_ids(ion_mode))
    document_ids = document_ids[50:]
    leftover_ids.sort()
    document_ids.sort()
    assert leftover_ids == document_ids


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents_none_missing(
    cleaned_data, saved_documents, documents_directory, s3_mock, spectra_stored
):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    ion_mode = "positive"

    documents = redis_dgw.list_missing_documents(
        document_ids=spectrum_ids_stored, ion_mode=ion_mode
    )

    assert len(documents) == 0


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents(cleaned_data, s3_mock, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]

    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    ion_mode = "positive"

    redis_dgw.remove_document_ids(spectrum_ids_stored, ion_mode=ion_mode)

    old_document_ids = [SpectrumDocument(sp) for sp in cleaned_data[:50]]
    redis_dgw.write_document_ids(old_document_ids, ion_mode=ion_mode)

    documents = redis_dgw.list_missing_documents(spectrum_ids_stored, ion_mode=ion_mode)

    assert len(documents) == 50
