import numpy as np
import pytest
from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum

from omigami.spectra_matching.ms2deepscore.helper_classes.spectrum_processor import (
    SpectrumProcessor,
)


@pytest.fixture
def spectrum_processor():
    return SpectrumProcessor()


@pytest.fixture
def single_spectrum(raw_spectra):
    return as_spectrum(raw_spectra[0])


@pytest.mark.slow
def test_process_spectra(positive_spectra_data, spectrum_processor):
    cleaned_data = spectrum_processor.process_spectra(positive_spectra_data, True)
    assert isinstance(cleaned_data[0], Spectrum)

    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data[0].get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data[0].get("charge"), int)
    assert cleaned_data[0].get("parent_mass")
    assert cleaned_data[0].get("spectrum_id")


def test_apply_ms2deepscore_filters(single_spectrum, spectrum_processor):
    mz_from = 10.0
    mz_to = 1000.0

    filtered_spectrum = spectrum_processor._apply_ms2deepscore_filters(single_spectrum)

    assert filtered_spectrum is not None
    assert all([mz_from <= mz <= mz_to for mz in filtered_spectrum.peaks.mz])


def test_apply_ms2deepscore_filters_not_enough_peaks(
    single_spectrum, spectrum_processor
):
    spectrum_with_not_enough_peaks = Spectrum(
        mz=single_spectrum.peaks.mz[:4],
        intensities=single_spectrum.peaks.intensities[:4],
    )
    filtered_spectrum = spectrum_processor._apply_ms2deepscore_filters(
        spectrum_with_not_enough_peaks
    )
    assert filtered_spectrum is None


@pytest.mark.skip("Uses internet connection.")
def test_run_missing_smiles_inchi_against_pubchem(
    common_cleaned_data, spectrum_processor
):
    cleaned_spectrum = spectrum_processor.process_spectra(
        [common_cleaned_data[2]], False
    )
    spectrum_with_inchikey = spectrum_processor._get_missing_inchis(cleaned_spectrum[0])

    assert not cleaned_spectrum[0].metadata.get("inchikey")
    assert len(spectrum_with_inchikey.metadata.get("inchikey")) > 13


def test_require_minimum_number_of_peaks(common_cleaned_data, spectrum_processor):
    assert spectrum_processor._require_minimum_number_of_peaks(
        common_cleaned_data[0], 5
    )
    mz = common_cleaned_data[0].peaks.mz
    spectrum_0_intensities = Spectrum(mz, np.zeros(len(mz)))

    assert not spectrum_processor._require_minimum_number_of_peaks(
        spectrum_0_intensities, 5
    )
