import os

import pytest
from matchms.Spectrum import Spectrum
from pytest_redis import factories

from omigami.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.storage import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
)

redis_db = factories.redisdb("redis_nooproc")

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)


def test_list_spectrum_ids(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    ids = dgw.list_spectrum_ids()
    assert len(ids) == len(spectrum_ids_stored)


def test_list_missing_spectra(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    spectrum_ids_stored += ["batman", "ROBEN"]

    dgw = RedisSpectrumDataGateway()
    spectra = dgw._list_missing_spectrum_ids("spectrum_data", spectrum_ids_stored)
    assert set(spectra) == {"batman", "ROBEN"}


def test_read_spectra(cleaned_data, spectra_stored):
    dgw = RedisSpectrumDataGateway()
    dgw._init_client()
    spectra = dgw.read_spectra()
    assert len(spectra) == len(cleaned_data)
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) > 0


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


def test_delete_spectrum_ids(spectra_stored):
    dgw = RedisSpectrumDataGateway()
    stored_ids = dgw.list_spectrum_ids()

    dgw.delete_spectra([stored_ids[0]])

    stored_ids_2 = dgw.list_spectrum_ids()

    assert set(stored_ids) - set(stored_ids_2) == {stored_ids[0]}
