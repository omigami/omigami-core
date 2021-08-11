import os
from typing import Iterable

import pytest
from matchms.Spectrum import Spectrum
from omigami.spec2vec.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    DOCUMENT_HASHES,
)
from omigami.spec2vec.entities.embedding import Embedding
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from pytest_redis import factories
from spec2vec.SpectrumDocument import SpectrumDocument

redis_db = factories.redisdb("redis_nooproc")


def test_write_spectrum_documents(tmpdir, cleaned_data):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]

    dgw = Spec2VecRedisSpectrumDataGateway()
    doc_dir = f"{tmpdir}/documents"

    if not os.path.exists(doc_dir):
        os.mkdir(doc_dir)

    dgw.write_spectrum_documents(spectrum_document_data, f"{doc_dir}/test.pckl")

    assert len(os.listdir(doc_dir)) == 1


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_spectrum_ids(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = Spec2VecRedisSpectrumDataGateway()
    ids = dgw.list_spectrum_ids()
    assert len(ids) == len(spectrum_ids_stored)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_spectra(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    spectrum_ids_stored += ["batman", "ROBEN"]

    dgw = Spec2VecRedisSpectrumDataGateway()
    spectra = dgw._list_missing_spectrum_ids("spectrum_data", spectrum_ids_stored)
    assert set(spectra) == {"batman", "ROBEN"}


def test_list_missing_documents(cleaned_data, tmpdir, write_documents):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = Spec2VecRedisSpectrumDataGateway()
    documents = dgw.list_missing_documents(spectrum_ids_stored, f"{tmpdir}/documents")

    assert len(documents) == 0


def test_read_spectra(cleaned_data, spectra_stored):
    dgw = Spec2VecRedisSpectrumDataGateway()
    spectra = dgw.read_spectra()
    assert len(spectra) == len(cleaned_data)
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) > 0


def test_read_documents(documents_data, tmpdir, write_documents):
    doc_dir = f"{tmpdir}/documents"

    dgw = Spec2VecRedisSpectrumDataGateway()
    documents = dgw.read_documents(doc_dir + "/test.pckl")

    assert len(documents) == len(documents_data)
    for document in documents:
        assert isinstance(document, SpectrumDocument)
        for word in document:
            assert word.startswith("peak@")


def test_read_embeddings(embeddings, embeddings_stored):
    dgw = Spec2VecRedisSpectrumDataGateway()
    embeddings_read = dgw.read_embeddings("positive", "1")
    assert len(embeddings_read) == len(embeddings)
    for embedding in embeddings_read:
        assert isinstance(embedding, Embedding)


def test_read_embeddings_within_range(embeddings, embeddings_stored, spectra_stored):
    dgw = Spec2VecRedisSpectrumDataGateway()
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
        assert isinstance(embedding, Embedding)
        assert (
            mz_min
            <= dgw.client.zscore(
                SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, embedding.spectrum_id
            )
            <= mz_max
        )


def test_read_spectra_ids_within_range(spectra_stored):
    dgw = Spec2VecRedisSpectrumDataGateway()
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


def test_delete_spectrum_ids(spectra_stored):
    dgw = Spec2VecRedisSpectrumDataGateway()
    stored_ids = dgw.list_spectrum_ids()

    dgw.delete_spectra([stored_ids[0]])

    stored_ids_2 = dgw.list_spectrum_ids()

    assert set(stored_ids) - set(stored_ids_2) == {stored_ids[0]}
