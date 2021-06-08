import os
from unittest.mock import MagicMock

import pytest
from prefect import Flow

from omigami.data_gateway import SpectrumDataGateway
from omigami.entities.spectrum_document import SpectrumDocumentData
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.tasks.process_spectrum import ProcessSpectrum
from omigami.tasks.process_spectrum.spectrum_processor import SpectrumProcessor
from omigami.test.conftest import TEST_TASK_CONFIG


def test_process_spectrum_task_calls(local_gnps_small_json, spectrum_ids):
    spectrum_gtw = MagicMock(spec=SpectrumDataGateway)
    input_gtw = FSInputDataGateway()
    spectrum_gtw.list_spectra_not_exist.side_effect = lambda x: x
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(
            local_gnps_small_json, spectrum_gtw, input_gtw, 2, False, **TEST_TASK_CONFIG
        )(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[process_task].result

    assert res.is_successful()
    assert set(data) == set(spectrum_ids)
    spectrum_gtw.list_spectra_not_exist.assert_not_called()
    spectrum_gtw.write_spectrum_documents.assert_called_once()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.slow
def test_process_spectrum_task(local_gnps_small_json, spectrum_ids):
    spectrum_gtw = RedisSpectrumDataGateway()
    spectrum_gtw.delete_spectra(spectrum_ids)
    input_gtw = FSInputDataGateway()
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(
            local_gnps_small_json, spectrum_gtw, input_gtw, 2, False, **TEST_TASK_CONFIG
        )(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[process_task].result

    assert set(data) == set(spectrum_ids)
    assert set(spectrum_gtw.list_spectrum_ids()) == set(spectrum_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.slow
def test_process_spectrum_task_map(local_gnps_small_json, spectrum_ids):
    spectrum_gtw = RedisSpectrumDataGateway()
    input_gtw = FSInputDataGateway()
    chunked_paths = [
        local_gnps_small_json,
        local_gnps_small_json,
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(
            local_gnps_small_json, spectrum_gtw, input_gtw, 2, False, **TEST_TASK_CONFIG
        ).map(chunked_paths)

    res = test_flow.run()
    data = res.result[process_task].result

    assert set(data[0]) == set(spectrum_ids)
    assert set(data[1]) == set(spectrum_ids)
    assert set(spectrum_gtw.list_spectrum_ids()) == set(spectrum_ids)


def test_clean_data(loaded_data):
    dc = SpectrumProcessor()

    cleaned_data = dc.create_documents(loaded_data)

    assert isinstance(cleaned_data[0], SpectrumDocumentData)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data[0].spectrum.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data[0].spectrum.get("charge"), int)
    assert cleaned_data[0].spectrum.get("parent_mass")
    assert cleaned_data[0].spectrum.get("spectrum_id")
