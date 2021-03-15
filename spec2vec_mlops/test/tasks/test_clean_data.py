from matchms import Spectrum

from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.load_data import DataLoader


def test_clean_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    cleaned_data = dc.clean_data(loaded_data)

    assert isinstance(cleaned_data, list)
    assert isinstance(cleaned_data[0], Spectrum)
