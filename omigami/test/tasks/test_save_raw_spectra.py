import pytest
import os

from matchms.importing.load_from_json import as_spectrum

from omigami.gateways import RedisSpectrumDataGateway
from omigami.gateways.input_data_gateway import FSInputDataGateway

from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters

from prefect import Flow


@pytest.fixture
def parameters(overwrite_all: bool = True):
    spectrum_dgw = RedisSpectrumDataGateway()
    input_dgw = FSInputDataGateway()
    parameters = SaveRawSpectraParameters(
        spectrum_dgw=spectrum_dgw, input_dgw=input_dgw
    )
    parameters.overwrite_all = overwrite_all
    return parameters


@pytest.fixture
def empty_database(parameters: SaveRawSpectraParameters, local_gnps_small_json):
    spectra = parameters.input_dgw.load_spectrum(local_gnps_small_json)
    parameters.spectrum_dgw.delete_spectra(
        [spec_id["spectrum_id"] for spec_id in spectra]
    )


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectrum_flow(parameters, local_gnps_small_json):
    """Tests if the task is flow ready"""
    # Setup Test
    empty_database(parameters, local_gnps_small_json)

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters).map(
            [local_gnps_small_json]
        )

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert len(data[0]) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_overwrite(parameters, local_gnps_small_json, spectra_stored):
    """Test if overwrite the task works by changing up the data"""
    # Setup Test
    loaded_data = parameters.input_dgw.load_spectrum(local_gnps_small_json)
    changed_value = loaded_data[0]["scan"]
    loaded_data[0]["scan"] = 5
    preserved_id = loaded_data[0]["spectrum_id"]

    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data]
    parameters.spectrum_dgw.write_raw_spectra(db_entries)

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=parameters)
    data = raw_spectra.run(local_gnps_small_json)

    # Test Results
    assert (
        parameters.spectrum_dgw.read_spectra([preserved_id])[preserved_id].metadata[
            "scan"
        ]
        != changed_value
    )
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert len(data) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_empty_db(parameters, local_gnps_small_json):
    """Test if new spectra get added to a empty database"""

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=parameters)
    data = raw_spectra.run(local_gnps_small_json)

    # Test Results
    assert len(data) == 100
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_add_new_spectra(parameters, local_gnps_small_json):
    """Test if new spectra get added to a database which already hosts some"""
    # Setup Test
    parameters.overwrite_all = False
    empty_database(parameters, local_gnps_small_json)

    loaded_data = parameters.input_dgw.load_spectrum(local_gnps_small_json)

    loaded_data[0]["scan"] = 5
    preserved_id = loaded_data[0]["spectrum_id"]

    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data[:30]]
    parameters.spectrum_dgw.write_raw_spectra(db_entries)

    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 30

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=parameters)
    data = raw_spectra.run(local_gnps_small_json)

    # Test Results
    assert len(data) == 100
    assert len(parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert (
        parameters.spectrum_dgw.read_spectra([preserved_id])[preserved_id].metadata[
            "scan"
        ]
        == 5
    )
