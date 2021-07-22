import os

import pytest
from matchms.importing.load_from_json import as_spectrum
from prefect import Flow

from omigami.gateways import RedisSpectrumDataGateway
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.spec2vec.config import SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters


@pytest.fixture
def create_parameters(overwrite_all_spectra: bool = True):
    spectrum_dgw = RedisSpectrumDataGateway()
    input_dgw = FSInputDataGateway()
    spectrum_cleaner = SpectrumCleaner()
    parameters = SaveRawSpectraParameters(
        spectrum_dgw=spectrum_dgw,
        input_dgw=input_dgw,
        spectrum_cleaner=spectrum_cleaner,
    )
    parameters.overwrite_all_spectra = overwrite_all_spectra
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
def test_write_raw_spectra(redis_db, loaded_data):
    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data]

    dgw = RedisSpectrumDataGateway()
    dgw.write_raw_spectra(db_entries)
    assert redis_db.zcard(SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET) == len(db_entries)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectum_flow(create_parameters, local_gnps_small_json):
    """Tests if the task if flow ready"""
    # Setup Test
    empty_database(create_parameters, local_gnps_small_json)

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=create_parameters).map(
            [local_gnps_small_json]
        )

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert len(data[0]) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_overwrite(
    create_parameters, local_gnps_small_json, spectra_stored
):
    """Test if overwrite the task wörks by changing up the data"""
    # Setup Test
    loaded_data = create_parameters.input_dgw.load_spectrum(local_gnps_small_json)
    changed_value = loaded_data[0]["scan"]
    loaded_data[0]["scan"] = 5
    preserved_id = loaded_data[0]["spectrum_id"]

    # Diana TODO: Call correct function
    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data]
    create_parameters.spectrum_dgw.write_raw_spectra(db_entries)

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=create_parameters)
    data = raw_spectra.run(local_gnps_small_json)

    # Test Results
    assert (
        create_parameters.spectrum_dgw.read_spectra([preserved_id])[0].get("scan")
        != changed_value
    )
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert len(data) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_empty_db(create_parameters, local_gnps_small_json):
    """Test if new spectra get added to a empty database"""

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=create_parameters)
    data = raw_spectra.run(local_gnps_small_json)

    # Test Results
    assert len(data) == 100
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_save_raw_spectra_add_new_spectra(create_parameters, local_gnps_small_json):
    """Test if new spectra get added to a database which already hosts some"""
    # Setup Test
    create_parameters.overwrite_all_spectra = False
    empty_database(create_parameters, local_gnps_small_json)

    loaded_data = create_parameters.input_dgw.load_spectrum(local_gnps_small_json)

    loaded_data[0]["ms_level"] = "5000000"
    preserved_id = loaded_data[0]["spectrum_id"]

    # Diana TODO: Call correct function
    db_entries = [as_spectrum(spectrum_data) for spectrum_data in loaded_data[:30]]
    create_parameters.spectrum_dgw.write_raw_spectra(db_entries)

    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 30

    # Run Functions
    raw_spectra = SaveRawSpectra(save_parameters=create_parameters)
    data = raw_spectra.run(local_gnps_small_json)

    spec = create_parameters.spectrum_dgw.read_spectra([preserved_id])

    # Test Results
    assert len(data) == 100
    assert len(create_parameters.spectrum_dgw.list_spectrum_ids()) == 100
    assert (
        create_parameters.spectrum_dgw.read_spectra([preserved_id])[0].get("ms_level")
        == "5000000"
    )
