from omigami.gateways import RedisSpectrumDataGateway
from omigami.spec2vec.config import PROJECT_NAME, DOCUMENT_HASHES
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways import Spec2VecFSDataGateway
from drfs.filesystems import get_fs
import pytest
import os


def test_write_spectrum_documents(documents_directory, cleaned_data, s3_mock):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]
    spectrum_document_data = [doc.document for doc in spectrum_document_data]
    spectrum_dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    dgw = Spec2VecFSDataGateway(spectrum_dgw)

    if not dgw.exists(documents_directory):
        dgw.makedir(documents_directory)

    dgw.serialize_documents(
        f"{documents_directory}/test.pickle",
        spectrum_document_data,
    )

    assert len(dgw.listdir(documents_directory)) == 1


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents(cleaned_data, s3_mock, spectra_stored):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    spectrum_dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    ion_mode = "None"
    dgw = Spec2VecFSDataGateway(spectrum_dgw, ion_mode)

    for doc_id in spectrum_ids_stored:
        spectrum_dgw.remove_document_id(doc_id, ion_mode=ion_mode)

    old_document_ids = spectrum_dgw.list_spectrum_ids()[:50]

    for doc_id in old_document_ids:
        spectrum_dgw.write_document_id(doc_id, ion_mode=ion_mode)

    documents = dgw.list_missing_documents(spectrum_ids_stored)

    assert len(documents) == 50


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_list_missing_documents_none_missing(
    cleaned_data, saved_documents, documents_directory, s3_mock, spectra_stored
):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    spectrum_dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    dgw = Spec2VecFSDataGateway(spectrum_dgw)

    documents = dgw.list_missing_documents(spectrum_ids_stored)

    assert len(documents) == 0


def test_read_documents_iter(saved_documents, documents_directory, s3_mock):
    fs = get_fs(documents_directory)
    document_file_names = fs.ls(documents_directory)
    spectrum_dgw = RedisSpectrumDataGateway(project=PROJECT_NAME)
    dgw = Spec2VecFSDataGateway(spectrum_dgw)

    document_counter = 0

    for doc in dgw.read_documents_iter(document_file_names):
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
