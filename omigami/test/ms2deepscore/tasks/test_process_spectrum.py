import pytest
from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum

from omigami.ms2deepscore.tasks.spectrum_processor import SpectrumProcessor


@pytest.fixture
def spectrum_processor():
    return SpectrumProcessor()


@pytest.fixture
def spectrum(loaded_data):
    return as_spectrum(loaded_data[0])


def test_process_spectra(loaded_data, spectrum_processor):
    cleaned_data = spectrum_processor.process_spectra(loaded_data)

    assert isinstance(cleaned_data[0], Spectrum)

    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data[0].get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data[0].get("charge"), int)
    assert cleaned_data[0].get("parent_mass")
    assert cleaned_data[0].get("spectrum_id")


def test_apply_ms2deepscore_filters(spectrum, spectrum_processor):
    mz_from = 10.0
    mz_to = 1000.0

    filtered_spectrum = spectrum_processor._apply_ms2deepscore_filters(spectrum)

    assert filtered_spectrum is not None
    assert all([mz_from <= mz <= mz_to for mz in filtered_spectrum.peaks.mz])

    spectrum_with_not_enough_peaks = Spectrum(
        mz=spectrum.peaks.mz[:4], intensities=spectrum.peaks.intensities[:4]
    )
    filtered_spectrum = spectrum_processor._apply_ms2deepscore_filters(
        spectrum_with_not_enough_peaks
    )
    assert filtered_spectrum is None
