from unittest.mock import Mock

import numpy as np
import pytest
from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum
from prefect import Flow

from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks.clean_raw_spectra import (
    SpectrumCleaner,
    CleanRawSpectra,
    CleanRawSpectraParameters,
)
from test.spectra_matching.conftest import ASSETS_DIR


def test_clean_raw_spectra_run(create_chunks_task):
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)
    t = CleanRawSpectra(fs_dgw, params)
    chunk_paths = create_chunks_task.run()

    res = t.run(chunk_paths[0])

    assert {sp.metadata["spectrum_id"] for sp in fs_dgw.read_from_file(res)}.issubset(
        {sp["SpectrumID"] for sp in fs_dgw.load_spectrum(chunk_paths[0])}
    )


def test_clean_raw_spectra_flow(create_chunks_task, mock_default_config):
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)

    with Flow("test-flow") as flow:
        cc = create_chunks_task()
        t = CleanRawSpectra(fs_dgw, params).map(cc)

    t._fs_dgw.load_spectrum = Mock(side_effect=fs_dgw.load_spectrum)
    res = flow.run()

    assert res.is_successful()
    assert t._fs_dgw.load_spectrum.call_count == 3  # there are three chunks processed

    # Running again to use cached results, and asserting it works
    t._fs_dgw.load_spectrum = Mock(side_effect=fs_dgw.load_spectrum)
    res_cached = flow.run()

    assert res_cached.is_successful()
    assert t._fs_dgw.load_spectrum.call_count == 0
    assert res.result[t].result == res_cached.result[t].result


@pytest.fixture
def single_spectrum(raw_spectra):
    return as_spectrum(raw_spectra[0])


@pytest.fixture
def spectrum_negative_intensity():
    return Spectrum(
        mz=np.sort(np.random.rand(216)), intensities=np.random.uniform(-2, 2, 216)
    )


def test_clean_data(single_spectrum):
    cleaned_data = SpectrumCleaner()._common_cleaning(single_spectrum)

    assert isinstance(cleaned_data, Spectrum)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data.get("charge"), int)
    assert cleaned_data.get("parent_mass")
    assert cleaned_data.get("spectrum_id")


def test_clean_data_negative_intensity(spectrum_negative_intensity):
    cleaned_data_negative_intensity = SpectrumCleaner()._common_cleaning(
        spectrum_negative_intensity
    )

    assert cleaned_data_negative_intensity is None


def test_clean_data_all_zero_intensities():
    spectrum_zero_intensity = Spectrum(
        mz=np.sort(np.random.rand(216)), intensities=np.zeros(216)
    )

    res = SpectrumCleaner()._common_cleaning(spectrum_zero_intensity)

    assert res is None


def test_apply_ms2deepscore_filters_negative_intensity(
    spectrum_negative_intensity, single_spectrum
):
    sc = SpectrumCleaner()
    removed_spectrum = sc._filter_negative_intensities(spectrum_negative_intensity)
    assert removed_spectrum is None

    spectrum = sc._filter_negative_intensities(single_spectrum)
    assert spectrum is not None

    negative_peak = spectrum.peaks
    negative_peak._intensities[4] = -10
    spectrum.peaks = negative_peak
    spectrum = sc._filter_negative_intensities(spectrum)
    assert spectrum is None
