from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum

from omigami.spectrum_cleaner import SpectrumCleaner


def test_clean_data(loaded_data):
    dc = SpectrumCleaner()

    cleaned_data = dc.basic_cleaning(as_spectrum(loaded_data[0]))

    assert isinstance(cleaned_data, Spectrum)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data.get("charge"), int)
    assert cleaned_data.get("parent_mass")
    assert cleaned_data.get("spectrum_id")
