import os
from unittest.mock import MagicMock

import pytest
from prefect import Flow

from omigami.data_gateway import SpectrumDataGateway
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway
from omigami.spec2vec.gateways.redis_spectrum_gateway import RedisSpectrumDataGateway
from omigami.spec2vec.tasks import ProcessSpectrumParameters
from omigami.spec2vec.tasks.process_spectrum import ProcessSpectrum
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)


def test_process_spectrum_calls(local_gnps_small_json, spectrum_ids):
    spectrum_gtw = MagicMock(spec=SpectrumDataGateway)
    input_gtw = FSInputDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, False)
    spectrum_gtw.list_existing_spectra.side_effect = lambda x: x
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(input_gtw, parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[process_task].result

    assert res.is_successful()
    assert set(data) == set(spectrum_ids)
    spectrum_gtw.list_existing_spectra.assert_not_called()
    spectrum_gtw.write_spectrum_documents.assert_called_once()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.slow
def test_process_spectrum(local_gnps_small_json, spectrum_ids, mock_default_config):
    spectrum_gtw = RedisSpectrumDataGateway()
    spectrum_gtw.delete_spectra(spectrum_ids)
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, False)
    input_gtw = FSInputDataGateway()
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(input_gtw, parameters)(local_gnps_small_json)

    res = test_flow.run()
    data = res.result[process_task].result

    assert set(data) == set(spectrum_ids)
    assert set(spectrum_gtw.list_spectrum_ids()) == set(spectrum_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
@pytest.mark.slow
def test_process_spectrum_map(local_gnps_small_json, spectrum_ids, mock_default_config):
    spectrum_gtw = RedisSpectrumDataGateway()
    input_gtw = FSInputDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, False)
    chunked_paths = [
        local_gnps_small_json,
        local_gnps_small_json,
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(input_gtw, parameters).map(chunked_paths)

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
