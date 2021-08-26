import os
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs
from prefect import Flow

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spec2vec.tasks import ProcessSpectrumParameters
from omigami.spec2vec.tasks.process_spectrum import ProcessSpectrum
from omigami.spec2vec.tasks.process_spectrum.spectrum_processor import (
    SpectrumProcessor,
)


def test_process_spectrum_calls(
    spectrum_ids, common_cleaned_data, s3_documents_directory, s3_mock
):
    redis_gateway = MagicMock(spec=Spec2VecRedisSpectrumDataGateway)
    data_gtw = MagicMock(spec=FSDataGateway)
    document_dgw_controller = MagicMock(spec=Spec2VecGatewayController)
    redis_gateway.read_spectra.return_value = common_cleaned_data

    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=s3_documents_directory,
        ion_mode="positive",
        n_decimals=2,
        overwrite_all_spectra=True,
    )

    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(data_gtw, document_dgw_controller, parameters)(
            spectrum_ids
        )

    res = test_flow.run()
    data = res.result[process_task].result

    assert res.is_successful()
    assert data == f"{s3_documents_directory}/documents0.pickle"
    redis_gateway.list_existing_spectra.assert_not_called()


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum(
    spectrum_ids, spectra_stored, mock_default_config, s3_documents_directory, s3_mock
):
    ion_mode = "positive"
    redis_gateway = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    document_dgw_controller = FSDataGateway()
    dgw_controller: Spec2VecGatewayController = Spec2VecGatewayController(
        redis_gateway, document_dgw_controller, ion_mode
    )
    # redis_gateway.delete_spectra(spectrum_ids[50:])
    # spectrum_ids = spectrum_ids[:50]

    documents_directory = f"{s3_documents_directory}/process_spectrum"
    fs = get_fs(documents_directory)
    fs.makedirs(documents_directory)

    redis_gateway.remove_document_ids(spectrum_ids, ion_mode)

    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=documents_directory,
        ion_mode=ion_mode,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(
            document_dgw_controller, dgw_controller, parameters
        )(spectrum_ids)

    res = test_flow.run()
    data = res.result[process_task].result

    assert data == f"{documents_directory}/documents0.pickle"


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_process_spectrum_map(
    spectrum_ids, spectra_stored, mock_default_config, s3_documents_directory
):
    ion_mode = "positive"
    redis_gateway = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    data_gtw = FSDataGateway()
    document_dgw_controller = Spec2VecGatewayController(
        redis_gateway, data_gtw, ion_mode
    )

    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=s3_documents_directory,
        ion_mode=ion_mode,
        n_decimals=2,
        overwrite_all_spectra=False,
    )
    chunked_paths = [
        spectrum_ids[:50],
        spectrum_ids[50:],
    ]
    with Flow("test-flow") as test_flow:
        process_task = ProcessSpectrum(
            data_gtw, document_dgw_controller, parameters
        ).map(chunked_paths)

    res = test_flow.run()
    data = res.result[process_task].result

    assert len(data) == 2
    assert set(redis_gateway.list_spectrum_ids()) == set(spectrum_ids)


def test_get_chunk_count(documents_stored, s3_documents_directory):
    ion_mode = "positive"
    redis_gateway = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    data_gtw = FSDataGateway()
    document_dgw_controller = Spec2VecGatewayController(
        redis_gateway, data_gtw, ion_mode
    )

    parameters = ProcessSpectrumParameters(
        spectrum_dgw=redis_gateway,
        documents_save_directory=s3_documents_directory,
        ion_mode=ion_mode,
        n_decimals=2,
        overwrite_all_spectra=False,
    )

    process_spectrum = ProcessSpectrum(data_gtw, document_dgw_controller, parameters)
    count = process_spectrum._get_chunk_count(s3_documents_directory)

    assert count == len(data_gtw.listdir(s3_documents_directory))


def test_clean_data(common_cleaned_data):
    dc = SpectrumProcessor()

    cleaned_data = dc.create_documents(common_cleaned_data)

    assert isinstance(cleaned_data[0], SpectrumDocumentData)
    assert cleaned_data[0].spectrum_id
    assert cleaned_data[0].document
