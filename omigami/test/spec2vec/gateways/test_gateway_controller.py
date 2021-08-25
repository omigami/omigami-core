import pytest
import os

from drfs.filesystems import get_fs

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.config import PROJECT_NAME
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_documents(documents_directory, cleaned_data, s3_mock):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    fs_dgw = FSDataGateway()

    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]
    spectrum_document_data = [doc.document for doc in spectrum_document_data]

    dgw_controller = Spec2VecGatewayController(
        redis_dgw,
        fs_dgw,
        ion_mode=ion_mode,
    )

    if not fs_dgw.exists(documents_directory):
        fs_dgw.makedirs(documents_directory)

    dgw_controller.write_documents(
        f"{documents_directory}/test.pickle",
        spectrum_document_data,
    )

    assert len(fs_dgw.listdir(documents_directory)) == 1
    assert len(redis_dgw.list_document_ids(ion_mode)) == len(cleaned_data)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_remove_documents(documents_directory, saved_documents, s3_mock):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    fs_dgw = FSDataGateway()

    dgw_controller = Spec2VecGatewayController(
        redis_dgw,
        fs_dgw,
        ion_mode=ion_mode,
    )
    for doc_path in fs_dgw.listdir(documents_directory):
        dgw_controller.remove_documents(doc_path)

    assert len(fs_dgw.listdir(documents_directory)) == 0
    assert len(redis_dgw.list_document_ids(ion_mode)) == 0
