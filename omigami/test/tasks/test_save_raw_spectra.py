import pytest
import os

from omigami.gateways import RedisSpectrumDataGateway
from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters

from prefect import Flow


@pytest.fixture
def create_parameters(overwrite_all: bool = True):
    spectrum_dgw = RedisSpectrumDataGateway()

    parameters = SaveRawSpectraParameters(spectrum_dgw=spectrum_dgw)
    parameters.overwrite_all = overwrite_all
    return parameters


@pytest.fixture
def empty_database(parameters: SaveRawSpectraParameters, spectra):
    parameters.spectrum_dgw.delete_spectra(
        [spec_id["spectrum_id"] for spec_id in spectra]
    )


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_overwrite(create_parameters, loaded_data):
    """Test if skipping the task wörks"""
    # Setup Test
    # TODO: Change up a spectra

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=create_parameters).map(loaded_data)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 0
    assert len(data) == 0


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_empty_db(create_parameters, loaded_data):
    """Test if new spectra get added to a empty database"""
    # Setup Test

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=create_parameters)(loaded_data)

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(data) == 100
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_add_new_spectra(create_parameters, loaded_data):
    """Test if new spectra get added to a database which already hosts some"""
    # Setup Test
    create_parameters.overwrite_all = False
    empty_database(create_parameters, loaded_data)

    spectra = create_parameters.input_dgw.load_spectrum(loaded_data)
    create_parameters.spectrum_dgw.write_raw_spectra(spectra[:30])

    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 30

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=create_parameters)
    data = raw_spectra.run()

    # Test Results
    assert len(data) == 100
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100
