import os
from unittest.mock import Mock

import pytest
from prefect import Flow

from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from omigami.spectra_matching.tasks import SaveCleanedSpectra


@pytest.fixture
def empty_database(spectrum_ids):
    RedisSpectrumDataGateway().delete_spectra(spectrum_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_cleaned_spectra_empty_db(cleaned_spectra_paths, empty_database):
    fs_dgw = FSDataGateway()
    expected_ids = {
        sp.metadata["spectrum_id"]
        for sp in fs_dgw.read_from_file(cleaned_spectra_paths[0])
    }
    spectrum_dgw = RedisSpectrumDataGateway()

    t = SaveCleanedSpectra(spectrum_dgw, fs_dgw)
    data = t.run(cleaned_spectra_paths[0])

    assert len(data) == 36  # 36 positive in the first chunk
    assert set(data) == expected_ids
    assert len(spectrum_dgw.list_spectrum_ids()) == 36


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_cleaned_spectra(cleaned_spectra_paths, cleaned_spectra, empty_database):
    """On this test some spectra will already be present in redis"""
    spectrum_dgw = RedisSpectrumDataGateway()
    first_36_spectra = cleaned_spectra[0]
    saved_spectra, new_spectra = first_36_spectra[:18], first_36_spectra[18:]
    spectrum_dgw.write_raw_spectra(saved_spectra)

    # We want to see if this method gets called with the remaining half of spectra
    spectrum_dgw.write_raw_spectra = Mock()
    t = SaveCleanedSpectra(spectrum_dgw, FSDataGateway())
    data = t.run(cleaned_spectra_paths[0])

    assert len(data) == 36
    spectrum_dgw.write_raw_spectra.assert_called_once_with(new_spectra)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_cleaned_spectrum_map(cleaned_spectra_paths, empty_database):
    spectrum_dgw = RedisSpectrumDataGateway()

    with Flow("test-flow") as test_flow:
        raw_spectra = SaveCleanedSpectra(
            RedisSpectrumDataGateway(), FSDataGateway()
        ).map(cleaned_spectra_paths)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    assert res.is_successful()
    # there are 63 positive spectra (out of 100 in the asset)
    assert len(spectrum_dgw.list_spectrum_ids()) == 63
    assert sum(len(d) for d in data) == 63
