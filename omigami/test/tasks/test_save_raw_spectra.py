import pytest

from omigami.ms2deepscore.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters
from omigami.gateways.input_data_gateway import FSInputDataGateway

from prefect import Flow


@pytest.skip
def create_parameters():
    spectrum_dgw = RedisSpectrumDataGateway()
    specturm_idgw = FSInputDataGateway()

    parameters = SaveRawSpectraParameters()
    parameters.spectrum_dgw = spectrum_dgw
    parameters.input_dgw = specturm_idgw

    return parameters


def test_save_raw_spectra_empty_db(local_gnps_small_json):
    # Setup Test
    parameters = create_parameters()

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert parameters.spectrum_dgw.list_spectrum_ids() == 100
    assert len(data) == 100


def test_save_raw_spectra_adding_new_spectra(local_gnps_small_json):
    # Setup Test
    parameters = create_parameters()
    spectra = parameters.input_dgw.load_spectrum(local_gnps_small_json)
    parameters.spectrum_dgw.write_raw_spectra(spectra[:30])

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert parameters.spectrum_dgw.list_spectrum_ids() == 100
    assert len(data) == 100


def test_save_raw_spectra_skip(local_gnps_small_json):
    # Setup Test
    parameters = create_parameters()
    parameters.skip_task = True
    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert parameters.spectrum_dgw.list_spectrum_ids() == 0
    assert len(data) == 0
