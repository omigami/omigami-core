from matchms import Spectrum

from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.load_data import DataLoader


def test_clean_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    for data in loaded_data:
        cleaned_data = dc.clean_data(data)

        assert isinstance(cleaned_data, Spectrum)
        # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
        assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
        assert isinstance(cleaned_data.get("charge"), int)
        assert cleaned_data.get("parent_mass")
        assert cleaned_data.get("spectrum_id")
