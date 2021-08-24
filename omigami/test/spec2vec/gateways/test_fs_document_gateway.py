from spec2vec import SpectrumDocument

from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways import Spec2VecFSDataGateway
from drfs.filesystems import get_fs
import pytest
import os

from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_spectrum_documents(documents_directory, cleaned_data, s3_mock):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]
    spectrum_document_data = [doc.document for doc in spectrum_document_data]
    dgw = Spec2VecGatewayController(ion_mode="positive")

    fs = get_fs(documents_directory)
    if not fs.exists(documents_directory):
        fs.makedirs(documents_directory)

    dgw.write_documents(
        f"{documents_directory}/test.pickle",
        spectrum_document_data,
    )

    assert len(fs.ls(documents_directory)) == 1


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents(cleaned_data, s3_mock, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]

    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    ion_mode = "positive"
    dgw_controller = Spec2VecRedisSpectrumDataGateway()

    redis_dgw.remove_document_ids(spectrum_ids_stored, ion_mode=ion_mode)

    old_document_ids = [SpectrumDocument(sp) for sp in cleaned_data[:50]]
    dgw_controller.write_document_ids(old_document_ids, ion_mode=ion_mode)

    documents = redis_dgw.list_missing_documents(spectrum_ids_stored, ion_mode=ion_mode)

    assert len(documents) == 50


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents_none_missing(
    cleaned_data, saved_documents, documents_directory, s3_mock, spectra_stored
):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    redis_dgw = Spec2VecRedisSpectrumDataGateway()
    ion_mode = "positive"

    documents = redis_dgw.list_missing_documents(
        document_ids=spectrum_ids_stored, ion_mode=ion_mode
    )

    assert len(documents) == 0


def test_read_documents_iter(saved_documents, documents_directory, s3_mock):
    fs = get_fs(documents_directory)
    document_file_names = fs.ls(documents_directory)
    dgw = Spec2VecFSDataGateway()

    document_counter = 0

    for doc in dgw.read_documents_iter(document_file_names):
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
