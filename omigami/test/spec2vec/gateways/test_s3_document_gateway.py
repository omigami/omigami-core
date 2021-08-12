from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData
from omigami.spec2vec.gateways import Spec2VecFSDocumentDataGateway
import os


def test_write_spectrum_documents(documents_directory, cleaned_data):
    spectrum_document_data = [
        SpectrumDocumentData(spectrum, 2) for spectrum in cleaned_data
    ]

    dgw = Spec2VecFSDocumentDataGateway()

    if not os.path.exists(documents_directory):
        os.mkdir(documents_directory)

    dgw.serialize_to_file(
        f"{documents_directory}/test.pickle",
        spectrum_document_data,
    )

    assert len(os.listdir(documents_directory)) == 1


def test_list_missing_documents(cleaned_data, save_documents, documents_directory):
    spectrum_ids_stored = [sp.metadata["spectrum_id"] for sp in cleaned_data]
    data_gtw = Spec2VecFSDocumentDataGateway()

    documents = data_gtw.list_missing_documents(
        spectrum_ids_stored, documents_directory
    )

    assert len(documents) == 0
