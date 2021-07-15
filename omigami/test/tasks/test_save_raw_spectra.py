import pytest

from omigami.spec2vec.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters
from omigami.gateways.input_data_gateway import FSInputDataGateway

from prefect import Flow


@pytest.mark.skip
def create_parameters(skip_task: bool):
    spectrum_dgw = RedisSpectrumDataGateway()
    specturm_idgw = FSInputDataGateway()

    parameters = SaveRawSpectraParameters(
        spectrum_dgw=spectrum_dgw, input_dgw=specturm_idgw
    )
    parameters.skip_task = skip_task
    return parameters


@pytest.mark.skip
def empty_database(parameters: SaveRawSpectraParameters, path: str):
    spectra = parameters.input_dgw.load_spectrum(path)
    parameters.spectrum_dgw.delete_spectra(
        [spec_id["spectrum_id"] for spec_id in spectra]
    )


def test_save_raw_spectra_skip(local_gnps_small_json):
    """Test if skipping the task wörks"""
    # Setup Test
    parameters = create_parameters(True)
    empty_database(parameters, local_gnps_small_json)

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 0
    assert len(data) == 0


def test_save_raw_spectra_empty_db(local_gnps_small_json):
    """Test if new spectra get added to an empty database"""
    # Setup Test
    parameters = create_parameters(False)

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(data) == 100
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100


def test_save_raw_spectra_adding_new_spectra(local_gnps_small_json):
    """Test if new spectra get added to an database which already hosts some"""
    # Setup Test
    parameters = create_parameters(False)
    empty_database(parameters, local_gnps_small_json)

    spectra = parameters.input_dgw.load_spectrum(local_gnps_small_json)
    parameters.spectrum_dgw.write_raw_spectra(spectra[:30])

    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 30

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(data) == 100
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100
