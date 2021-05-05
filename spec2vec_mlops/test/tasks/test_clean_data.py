from unittest.mock import MagicMock

from matchms import Spectrum
from prefect import Flow

from spec2vec_mlops.tasks.clean_data import CleanData
from spec2vec_mlops.tasks.clean_data.clean_data import DataCleaner
from spec2vec_mlops.tasks.data_gateway import SpectrumDataGateway


def test_clean_data_task(loaded_data):
    gnps = list(range(50))
    spectrum_gtw = MagicMock(spec=SpectrumDataGateway)
    spectrum_gtw.load_gnps.return_value = gnps
    with Flow("test-flow") as test_flow:
        clean_task = CleanData(spectrum_gtw)(chunk_size=10)

    res = test_flow.run()
    data = res.result[clean_task].result

    assert res.is_successful()
    assert len(data) == 5
    assert set(data[0]) == set(range(10))
    spectrum_gtw.load_gnps.assert_called_once()


def test_clean_data(loaded_data):
    dc = DataCleaner()

    for data in loaded_data:
        cleaned_data = dc.clean_data(data)

        assert isinstance(cleaned_data, Spectrum)
        # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
        assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
        assert isinstance(cleaned_data.get("charge"), int)
        assert cleaned_data.get("parent_mass")
        assert cleaned_data.get("spectrum_id")
