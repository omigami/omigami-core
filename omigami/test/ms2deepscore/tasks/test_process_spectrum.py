from matchms import Spectrum

from omigami.ms2deepscore.tasks.spectrum_processor import SpectrumProcessor


def test_process_spectra(loaded_data):
    spectrum_processor = SpectrumProcessor()

    cleaned_data = spectrum_processor.process_spectra(loaded_data)

    assert isinstance(cleaned_data[0], Spectrum)

    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data[0].get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data[0].get("charge"), int)
    assert cleaned_data[0].get("parent_mass")
    assert cleaned_data[0].get("spectrum_id")
