import os

import pytest
from matchms.Spectrum import Spectrum
from matchms.importing.load_from_json import as_spectrum
from pytest_redis import factories

from omigami.config import SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, EMBEDDING_HASHES
from omigami.spectra_matching.spec2vec import SPEC2VEC_PROJECT_NAME
from omigami.spectra_matching.storage import RedisSpectrumDataGateway

redis_db = factories.redisdb("redis_nooproc")

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
_PROJECT = "project"


def test_list_spectrum_ids(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    dgw = RedisSpectrumDataGateway(project=SPEC2VEC_PROJECT_NAME)
    ids = dgw.list_spectrum_ids()
    assert len(ids) == len(spectrum_ids_stored)


def test_list_missing_spectra(cleaned_data, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    spectrum_ids_stored += ["batman", "ROBEN"]

    dgw = RedisSpectrumDataGateway(_PROJECT)
    spectra = dgw._list_missing_spectrum_ids("spectrum_data", spectrum_ids_stored)
    assert set(spectra) == {"batman", "ROBEN"}


def test_read_spectra(cleaned_data, spectra_stored):
    dgw = RedisSpectrumDataGateway(_PROJECT)
    dgw._init_client()
    spectra = dgw.read_spectra()
    assert len(spectra) == len(cleaned_data)
    for spectrum in spectra:
        assert isinstance(spectrum, Spectrum)
        assert len(spectrum.peaks) > 0


def test_read_spectra_ids_within_range(spectra_stored):
    dgw = RedisSpectrumDataGateway(_PROJECT)
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
    dgw = RedisSpectrumDataGateway(_PROJECT)
    stored_ids = dgw.list_spectrum_ids()

    dgw.delete_spectra([stored_ids[0]])

    stored_ids_2 = dgw.list_spectrum_ids()

    assert set(stored_ids) - set(stored_ids_2) == {stored_ids[0]}


def test_write_raw_spectra(redis_db, loaded_data):
    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data]

    dgw = RedisSpectrumDataGateway(_PROJECT)
    dgw.write_raw_spectra(db_entries)
    assert redis_db.zcard(SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET) == len(db_entries)


def test_delete_embeddings(redis_db, ms2deepscore_embeddings_stored):
    dgw = RedisSpectrumDataGateway("ms2deepscore")
    hash_keys = redis_db.scan()[1]
    embeddings_key = dgw._format_redis_key(EMBEDDING_HASHES, "positive").encode()
    # asserting for correct test setup
    assert embeddings_key in hash_keys

    dgw.delete_embeddings("positive")
    updated_hkeys = redis_db.scan()[1]

    assert embeddings_key not in updated_hkeys
    # Asserting deleting the key still works even if the key is not there
    dgw.delete_embeddings("positive")
