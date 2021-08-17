from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways import Spec2VecFSDataGateway
import os


def test_write_spectrum_documents(documents_directory, cleaned_data):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]

    dgw = Spec2VecFSDataGateway()

    if not os.path.exists(documents_directory):
        os.mkdir(documents_directory)

    dgw.serialize_to_file(
        f"{documents_directory}/test.pickle",
        spectrum_document_data,
    )

    assert len(os.listdir(documents_directory)) == 1


def test_list_missing_documents(cleaned_data, saved_documents, documents_directory):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    data_gtw = Spec2VecFSDataGateway()

    document_file_names = os.listdir(documents_directory)

    for file in document_file_names[:5]:
        os.remove(f"{documents_directory}/{file}")

    documents = data_gtw.list_missing_documents(
        spectrum_ids_stored, documents_directory
    )

    assert len(documents) == 50


def test_list_missing_documents_none_missing(
    cleaned_data, saved_documents, documents_directory
):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    data_gtw = Spec2VecFSDataGateway()

    documents = data_gtw.list_missing_documents(
        spectrum_ids_stored, documents_directory
    )

    assert len(documents) == 0


def test_read_documents_iter(saved_documents, documents_directory):
    document_file_names = os.listdir(documents_directory)
    data_gtw = Spec2VecFSDataGateway()
    document_counter = 0

    document_paths = [f"{documents_directory}/{file}" for file in document_file_names]

    for doc in data_gtw.read_documents_iter(document_paths):
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
