import numpy as np
import pytest
from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum

from omigami.spectra_matching.spectrum_cleaner import SpectrumCleaner


@pytest.fixture
def spectrum_cleaner():
    return SpectrumCleaner()


@pytest.fixture
def spectrum(loaded_data):
    return as_spectrum(loaded_data[0])


@pytest.fixture
def spectrum_negative_intensity():
    return Spectrum(
        mz=np.sort(np.random.rand(216)), intensities=np.random.uniform(-2, 2, 216)
    )


def test_clean_data(spectrum, spectrum_cleaner, spectrum_negative_intensity):

    cleaned_data = spectrum_cleaner._common_cleaning(spectrum)
    cleaned_data_negative_intensity = spectrum_cleaner._common_cleaning(
        spectrum_negative_intensity
    )
    assert cleaned_data_negative_intensity is None
    assert isinstance(cleaned_data, Spectrum)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data.get("charge"), int)
    assert cleaned_data.get("parent_mass")
    assert cleaned_data.get("spectrum_id")


def test_apply_ms2deepscore_filters_negative_intensity(
    spectrum_negative_intensity, spectrum, spectrum_cleaner
):
    spectrum_negative_intensity = spectrum_cleaner._filter_negative_intensities(
        spectrum_negative_intensity
    )
    spectrum = spectrum_cleaner._filter_negative_intensities(spectrum)

    assert spectrum_negative_intensity is None
    assert spectrum is not None
