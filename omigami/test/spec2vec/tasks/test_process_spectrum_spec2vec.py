import os
from unittest.mock import MagicMock

import pytest
from prefect import Flow

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spec2vec.tasks import ProcessSpectrumParameters
from omigami.spec2vec.tasks.process_spectrum import ProcessSpectrum
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)


def test_process_spectrum_calls(spectrum_ids, common_cleaned_data):
    spectrum_gtw = MagicMock(spec=Spec2VecRedisSpectrumDataGateway)
    spectrum_gtw.read_spectra.return_value = common_cleaned_data
    data_gtw = FSDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, True)

    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters)(spectrum_ids)

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
def test_process_spectrum(spectrum_ids, spectra_stored, mock_default_config):
    spectrum_gtw = Spec2VecRedisSpectrumDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, False)
    data_gtw = FSDataGateway()
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters)(spectrum_ids)

    res = test_flow.run()
    data = res.result[process_task].result

    assert set(data) == set(spectrum_ids)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum_map(spectrum_ids, spectra_stored, mock_default_config):
    spectrum_gtw = Spec2VecRedisSpectrumDataGateway()
    data_gtw = FSDataGateway()
    parameters = ProcessSpectrumParameters(spectrum_gtw, 2, False)
    chunked_paths = [
        spectrum_ids[:50],
        spectrum_ids[50:],
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters).map(chunked_paths)

    res = test_flow.run()
    data = res.result[process_task].result

    assert set(data[0]) == set(spectrum_ids[:50])
    assert set(data[1]) == set(spectrum_ids[50:])
    assert set(spectrum_gtw.list_spectrum_ids()) == set(spectrum_ids)


def test_clean_data(common_cleaned_data):
    dc = SpectrumProcessor()

    cleaned_data = dc.create_documents(common_cleaned_data)

    assert isinstance(cleaned_data[0], SpectrumDocumentData)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data[0].spectrum.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data[0].spectrum.get("charge"), int)
    assert cleaned_data[0].spectrum.get("parent_mass")
    assert cleaned_data[0].spectrum.get("spectrum_id")
