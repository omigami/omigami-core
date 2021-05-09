from unittest.mock import MagicMock

from matchms import Spectrum
from prefect import Flow

from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrum
from spec2vec_mlops.tasks.process_spectrum.spectrum_processor import SpectrumProcessor
from spec2vec_mlops.tasks.data_gateway import SpectrumDataGateway
from spec2vec_mlops.test.conftest import TEST_TASK_CONFIG


def test_process_spectrum_task(local_gnps_small_json, spectrum_ids):
    spectrum_gtw = MagicMock(spec=SpectrumDataGateway)
    input_gtw = FSInputDataGateway()
    spectrum_gtw.list_spectra_not_exist.side_effect = lambda x: x
    with Flow("test-flow") as test_flow:
        clean_task = ProcessSpectrum(
            spectrum_gtw, input_gtw, local_gnps_small_json, 2, False, **TEST_TASK_CONFIG
        )(spectrum_ids[:10])

    res = test_flow.run()
    data = res.result[clean_task].result

    assert res.is_successful()
    assert len(data) == len(spectrum_ids[:10])
    spectrum_gtw.list_spectra_not_exist.assert_not_called()
    spectrum_gtw.write_spectrum_documents.assert_called_once()


def test_clean_data(loaded_data):
    dc = SpectrumProcessor()

    for data in loaded_data:
        cleaned_data = dc.process_data(data)

        assert isinstance(cleaned_data, Spectrum)
        # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
        assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
        assert isinstance(cleaned_data.get("charge"), int)
        assert cleaned_data.get("parent_mass")
        assert cleaned_data.get("spectrum_id")
