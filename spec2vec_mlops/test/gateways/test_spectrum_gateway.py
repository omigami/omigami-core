import os
import pickle
from typing import Iterable

import pytest
from matchms.Spectrum import Spectrum
from pytest_redis import factories
from spec2vec.SpectrumDocument import SpectrumDocument

from spec2vec_mlops.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    DOCUMENT_HASHES,
    EMBEDDING_HASHES,
)
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway


redis_db = factories.redisdb("redis_nooproc")


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)


@pytest.fixture()
def spectra_stored(redis_db, cleaned_data):
    pipe = redis_db.pipeline()
    for spectrum in cleaned_data:
        pipe.zadd(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
            {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
        )
        pipe.hset(
            SPECTRUM_HASHES, spectrum.metadata["spectrum_id"], pickle.dumps(spectrum)
        )
    pipe.execute()


@pytest.fixture()
def documents_stored(redis_db, cleaned_data, documents_data):
    pipe = redis_db.pipeline()
    for i, document in enumerate(documents_data):
        pipe.hset(
            DOCUMENT_HASHES,
            cleaned_data[i].metadata["spectrum_id"],
            pickle.dumps(document),
        )
    pipe.execute()


@pytest.fixture()
def embeddings_stored(redis_db, embeddings):
    run_id = "1"
    pipe = redis_db.pipeline()
    for embedding in embeddings:
        pipe.hset(
            f"{EMBEDDING_HASHES}_{run_id}",
            embedding.spectrum_id,
            pickle.dumps(embedding),
        )
    pipe.execute()


def test_write_spectrum_documents(redis_db, cleaned_data):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]

    dgw = RedisSpectrumDataGateway()
    dgw.write_spectrum_documents(spectrum_document_data)

    assert redis_db.zcard(SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET) == len(cleaned_data)
    assert redis_db.hlen(SPECTRUM_HASHES) == len(cleaned_data)
    assert redis_db.hlen(DOCUMENT_HASHES) == len(cleaned_data)


def test_list_spectrum_ids(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = RedisSpectrumDataGateway()
    ids = dgw.list_spectrum_ids()
    assert len(ids) == len(spectrum_ids_stored)


def test_list_spectra_not_exist(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = RedisSpectrumDataGateway()
    spectra = dgw.list_spectra_not_exist(spectrum_ids_stored)
    assert len(spectra) == 0


def test_list_documents_not_exist(cleaned_data, documents_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = RedisSpectrumDataGateway()
    documents = dgw.list_documents_not_exist(spectrum_ids_stored)
    assert len(documents) == 0


def test_read_spectra(cleaned_data, spectra_stored):
    dgw = RedisSpectrumDataGateway()
    spectra = dgw.read_spectra()
    assert len(spectra) == len(cleaned_data)
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) > 0


def test_read_documents(documents_data, documents_stored):
    dgw = RedisSpectrumDataGateway()
    documents = dgw.read_documents()
    assert len(documents) == len(documents_data)
    for document in documents:
        assert isinstance(document, SpectrumDocument)
        for word in document:
            assert word.startswith("peak@")


def test_read_embeddings(embeddings, embeddings_stored):
    dgw = RedisSpectrumDataGateway()
    embeddings_read = dgw.read_embeddings("1")
    assert len(embeddings_read) == len(embeddings)
    for embedding in embeddings_read:
        assert isinstance(embedding, Embedding)


def test_read_embeddings_within_range(embeddings, embeddings_stored, spectra_stored):
    dgw = RedisSpectrumDataGateway()
    dgw._init_client()
    mz_min = 300
    mz_max = 600
    filtered_spectra = dgw.client.zrangebyscore(
        SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, mz_min, mz_max
    )
    spectrum_ids_within_range = dgw.get_spectrum_ids_within_range(mz_min, mz_max)
    embeddings_read = dgw.read_embeddings("1", spectrum_ids_within_range)
    assert len(embeddings_read) == len(filtered_spectra)
    for embedding in embeddings_read:
        assert isinstance(embedding, Embedding)
        assert (
            mz_min
            <= dgw.client.zscore(
                SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, embedding.spectrum_id
            )
            <= mz_max
        )


def test_read_spectra_ids_within_range(spectra_stored):
    dgw = RedisSpectrumDataGateway()
    dgw._init_client()
    mz_min = 300
    mz_max = 600
    filtered_spectra = dgw.client.zrangebyscore(
        SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, mz_min, mz_max
    )
    spectra_ids_within_range = dgw.get_spectrum_ids_within_range(mz_min, mz_max)
    assert len(spectra_ids_within_range) == len(filtered_spectra)
    for spectrum_id in spectra_ids_within_range:
        assert (
            mz_min
            <= dgw.client.zscore(SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, spectrum_id)
            <= mz_max
        )


def test_read_documents_iter(documents_stored):
    dgw = RedisSpectrumDataGateway()
    doc_iter = dgw.read_documents_iter()
    assert isinstance(doc_iter, Iterable)

    all_sentences = 0
    all_words = 0
    for sentence in doc_iter:
        all_sentences += 1
        assert isinstance(sentence, SpectrumDocument)
        for word in sentence:
            all_words += 1
            assert word.startswith("peak@")

    all_words_no_iterator = 0
    for spectrum in dgw.read_documents():
        all_words_no_iterator += len(spectrum)
    assert all_words == all_words_no_iterator


def test_delete_spectrum_ids(spectra_stored):
    dgw = RedisSpectrumDataGateway()
    stored_ids = dgw.list_spectrum_ids()

    dgw.delete_spectra([stored_ids[0]])

    stored_ids_2 = dgw.list_spectrum_ids()

    assert set(stored_ids) - set(stored_ids_2) == {stored_ids[0]}
