import os

import pytest

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.spectrum_document import SpectrumDocumentDataGateway


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_documents(s3_documents_directory, documents_data, s3_mock):
    ion_mode = "positive"
    fs_dgw = FSDataGateway()

    document_dgw = SpectrumDocumentDataGateway(ion_mode, fs_dgw)

    if not fs_dgw.exists(s3_documents_directory):
        fs_dgw.makedirs(s3_documents_directory)

    document_dgw.write_documents(
        f"{s3_documents_directory}/test.pickle",
        documents_data,
    )

    assert len(fs_dgw.listdir(s3_documents_directory)) == 1
    assert len(document_dgw.list_document_ids(ion_mode)) == len(documents_data)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_remove_documents(s3_documents_directory, documents_stored, s3_mock):
    ion_mode = "positive"
    fs_dgw = FSDataGateway()

    document_dgw = SpectrumDocumentDataGateway(
        ion_mode,
        fs_dgw,
    )
    for doc_path in fs_dgw.listdir(s3_documents_directory):
        document_dgw.remove_documents_file(doc_path)

    assert len(fs_dgw.listdir(s3_documents_directory)) == 0
    assert len(document_dgw.list_document_ids(ion_mode)) == 0
