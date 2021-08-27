import os

import pytest
from spec2vec import SpectrumDocument

from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_document_ids(documents_stored):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()

    stored_ids = redis_dgw.list_document_ids(ion_mode)

    assert len(documents_stored) == len(stored_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_document_ids(documents_data):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()

    redis_dgw.write_document_ids(documents_data, ion_mode)

    stored_ids = list(redis_dgw.list_document_ids(ion_mode))
    ids = [doc.get("spectrum_id") for doc in documents_data]
    stored_ids.sort()
    ids.sort()
    assert stored_ids == ids


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_remove_document_ids(documents_stored):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway()

    document_ids = [doc.get("spectrum_id") for doc in documents_stored]

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
    documents_data, documents_stored, spectra_stored
):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in documents_data]
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
def test_list_missing_documents(documents_data, s3_mock, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in documents_data]

    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    ion_mode = "positive"

    redis_dgw.remove_document_ids(spectrum_ids_stored, ion_mode=ion_mode)

    old_document_ids = [SpectrumDocument(sp) for sp in documents_data[:50]]
    redis_dgw.write_document_ids(old_document_ids, ion_mode=ion_mode)

    documents = redis_dgw.list_missing_documents(spectrum_ids_stored, ion_mode=ion_mode)

    assert len(documents) == 50
