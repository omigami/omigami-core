from matchms import Spectrum

from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.load_data import DataLoader


def test_clean_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    cleaned_data = dc.clean_data(loaded_data)

    assert isinstance(cleaned_data, list)
    assert all(isinstance(spec, Spectrum) for spec in cleaned_data)
    assert all(spec.get("inchi") != "N/A" for spec in cleaned_data)
    assert all(isinstance(spec.get("charge"), int) for spec in cleaned_data)
