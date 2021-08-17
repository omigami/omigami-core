import os
from unittest.mock import MagicMock

import pytest
from prefect import Flow

from omigami.gateways import RedisSpectrumDataGateway


from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.fs_document_gateway import (
    Spec2VecFSDataGateway,
)

from omigami.spec2vec.tasks import ProcessSpectrumParameters
from omigami.spec2vec.tasks.process_spectrum import ProcessSpectrum
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)


def test_process_spectrum_calls(spectrum_ids, common_cleaned_data, documents_directory):
    spectrum_gtw = MagicMock(spec=RedisSpectrumDataGateway)
    spectrum_gtw.read_spectra.return_value = common_cleaned_data
    data_gtw = MagicMock(spec=Spec2VecFSDataGateway)
    parameters = ProcessSpectrumParameters(
        spectrum_dgw=spectrum_gtw,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=True,
    )

    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters)(spectrum_ids)

    res = test_flow.run()
    data = res.result[process_task].result

    assert res.is_successful()
    assert data == f"{documents_directory}/documents0.pkl"
    spectrum_gtw.list_existing_spectra.assert_not_called()
    data_gtw.serialize_to_file.assert_called_once()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum(
    spectrum_ids, spectra_stored, mock_default_config, documents_directory
):
    spectrum_gtw = RedisSpectrumDataGateway()
    parameters = ProcessSpectrumParameters(
        spectrum_dgw=spectrum_gtw,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    os.mkdir(documents_directory)
    data_gtw = Spec2VecFSDataGateway()
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters)(spectrum_ids)

    res = test_flow.run()
    data = res.result[process_task].result

    assert data == f"{documents_directory}/documents0.pkl"


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum_map(
    spectrum_ids, spectra_stored, mock_default_config, documents_directory
):
    spectrum_gtw = RedisSpectrumDataGateway()
    data_gtw = Spec2VecFSDataGateway()
    parameters = ProcessSpectrumParameters(
        spectrum_dgw=spectrum_gtw,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=False,
    )
    chunked_paths = [
        spectrum_ids[:50],
        spectrum_ids[50:],
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters).map(chunked_paths)

    res = test_flow.run()
    data = res.result[process_task].result

    assert len(data) == 2
    assert set(spectrum_gtw.list_spectrum_ids()) == set(spectrum_ids)


# TODO: Will remove if CI tests pass
# def test_clean_data(common_cleaned_data):
#    dc = SpectrumProcessor()
#
#    cleaned_data = dc.create_documents(common_cleaned_data)
#
#    assert isinstance(cleaned_data[0], SpectrumDocumentData)
# Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
#    assert cleaned_data[0].spectrum.get("inchi") not in ["N/A", "NA", "n/a", None]
#    assert isinstance(cleaned_data[0].spectrum.get("charge"), int)
#    assert cleaned_data[0].spectrum.get("parent_mass")
#    assert cleaned_data[0].spectrum.get("spectrum_id")


def test_get_chunk_count(saved_documents, documents_directory):
    data_gtw = Spec2VecFSDataGateway()
    redis_gateway = RedisSpectrumDataGateway()
    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    process_spectrum = ProcessSpectrum(data_gtw, parameters)
    count = process_spectrum._get_chunk_count(documents_directory)

    assert count == len(saved_documents[0])
