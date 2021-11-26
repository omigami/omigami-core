import os

import pytest
from spec2vec import SpectrumDocument

from omigami.config import SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET
from omigami.spectra_matching.spec2vec.config import PROJECT_NAME
from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.spec2vec.gateways.redis_spectrum_document import (
    RedisSpectrumDocumentDataGateway,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_document_ids(documents_stored):
    ion_mode = "positive"
    redis_dgw = RedisSpectrumDocumentDataGateway()

    stored_ids = redis_dgw.list_document_ids(ion_mode)

    assert len(documents_stored) == len(stored_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_document_ids(documents_data):
    ion_mode = "positive"
    redis_dgw = RedisSpectrumDocumentDataGateway()

    redis_dgw.write_documents(documents_data, ion_mode)

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
    redis_dgw = RedisSpectrumDocumentDataGateway()

    document_ids = [doc.get("spectrum_id") for doc in documents_stored]

    redis_dgw.remove_documents(document_ids[:50], ion_mode)

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
    redis_dgw = RedisSpectrumDocumentDataGateway()
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

    redis_dgw = RedisSpectrumDocumentDataGateway()
    ion_mode = "positive"

    redis_dgw.remove_documents(spectrum_ids_stored, ion_mode=ion_mode)

    old_document_ids = [SpectrumDocument(sp) for sp in documents_data[:50]]
    redis_dgw.write_documents(old_document_ids, ion_mode=ion_mode)

    documents = redis_dgw.list_missing_documents(spectrum_ids_stored, ion_mode=ion_mode)

    assert len(documents) == 50


def test_read_embeddings(spec2vec_embeddings, spec2vec_embeddings_stored):
    dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    dgw._init_client()
    embeddings_read = dgw.read_embeddings("positive", "1")
    assert len(embeddings_read) == len(spec2vec_embeddings)
    for embedding in embeddings_read:
        assert isinstance(embedding, Spec2VecEmbedding)


def test_read_embeddings_within_range(
    spec2vec_embeddings, spec2vec_embeddings_stored, spectra_stored
):
    dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    dgw._init_client()
    mz_min = 300
    mz_max = 600
    filtered_spectra = dgw.client.zrangebyscore(
        SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, mz_min, mz_max
    )
    spectrum_ids_within_range = dgw.get_spectrum_ids_within_range(mz_min, mz_max)
    embeddings_read = dgw.read_embeddings("positive", "1", spectrum_ids_within_range)
    assert len(embeddings_read) == len(filtered_spectra)
    for embedding in embeddings_read:
        assert isinstance(embedding, Spec2VecEmbedding)
        assert (
            mz_min
            <= dgw.client.zscore(
                SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, embedding.spectrum_id
            )
            <= mz_max
        )
