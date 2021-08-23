import os
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME

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
    assert data == f"{documents_directory}/documents0.pickle"
    spectrum_gtw.list_existing_spectra.assert_not_called()
    data_gtw.serialize_documents.assert_called_once()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum(
    spectrum_ids, spectra_stored, mock_default_config, documents_directory, s3_mock
):
    spectrum_gtw = RedisSpectrumDataGateway(PROJECT_NAME)
    data_gtw = Spec2VecFSDataGateway(spectrum_gtw)

    spectrum_gtw.delete_spectra(spectrum_ids[50:])
    spectrum_ids = spectrum_ids[:50]

    documents_directory = f"{documents_directory}/process_spectrum"
    fs = get_fs(documents_directory)
    fs.makedirs(documents_directory)

    for doc_id in spectrum_ids[:50]:
        spectrum_gtw.remove_document_id(doc_id, data_gtw._ion_mode)

    parameters = ProcessSpectrumParameters(
        spectrum_dgw=spectrum_gtw,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, parameters)(spectrum_ids)

    res = test_flow.run()
    data = res.result[process_task].result

    assert data == f"{documents_directory}/documents0.pickle"


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum_map(
    spectrum_ids, spectra_stored, mock_default_config, documents_directory
):
    spectrum_gtw = RedisSpectrumDataGateway(PROJECT_NAME)
    data_gtw = Spec2VecFSDataGateway(spectrum_gtw)
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


def test_get_chunk_count(saved_documents, documents_directory):

    redis_gateway = RedisSpectrumDataGateway(PROJECT_NAME)
    data_gtw = Spec2VecFSDataGateway(redis_gateway)
    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=documents_directory,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    process_spectrum = ProcessSpectrum(data_gtw, parameters)
    count = process_spectrum._get_chunk_count(documents_directory)

    assert count == len(saved_documents[0])


def test_clean_data(common_cleaned_data):
    dc = SpectrumProcessor()

    cleaned_data = dc.create_documents(common_cleaned_data)

    assert isinstance(cleaned_data[0], SpectrumDocumentData)
    assert cleaned_data[0].spectrum_id
    assert cleaned_data[0].document
