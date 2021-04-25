import os
import pickle
from typing import Iterable

import pytest
from matchms.Spectrum import Spectrum
from pytest_redis import factories
from spec2vec.SpectrumDocument import SpectrumDocument

from spec2vec_mlops import config
from spec2vec_mlops.entities.embedding import Embedding
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.gateways.redis_gateway import RedisDataGateway

SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["redis"]["spectrum_id_sorted_set"]
SPECTRUM_HASHES = config["redis"]["spectrum_hashes"]
DOCUMENT_HASHES = config["redis"]["document_hashes"]
EMBEDDING_HASHES = config["redis"]["embedding_hashes"]

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

    dgw = RedisDataGateway()
    dgw.write_spectrum_documents(spectrum_document_data)

    assert redis_db.zcard(SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET) == len(cleaned_data)
    assert redis_db.hlen(SPECTRUM_HASHES) == len(cleaned_data)
    assert redis_db.hlen(DOCUMENT_HASHES) == len(cleaned_data)


def test_read_spectra(cleaned_data, spectra_stored):
    dgw = RedisDataGateway()
    spectra = dgw.read_spectra()
    assert len(spectra) == len(cleaned_data)
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) > 0


def test_read_documents(documents_data, documents_stored):
    dgw = RedisDataGateway()
    documents = dgw.read_documents()
    assert len(documents) == len(documents_data)
    for document in documents:
        assert isinstance(document, SpectrumDocument)
        for word in document:
            assert word.startswith("peak@")


def test_read_embeddings(embeddings, embeddings_stored):
    dgw = RedisDataGateway()
    embeddings_read = dgw.read_embeddings("1")
    assert len(embeddings_read) == len(embeddings)
    for embedding in embeddings_read:
        assert isinstance(embedding, Embedding)


def test_read_documents_iter(documents_stored):
    dgw = RedisDataGateway()
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
