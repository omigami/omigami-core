from omigami.spec2vec.gateways import Spec2VecFSDataGateway
from drfs.filesystems import get_fs
import pytest
import os

from omigami.spec2vec.gateways.fs_document_gateway import FileSystemDocumentIterator


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_read_documents_iter(saved_documents, documents_directory, s3_mock):
    fs = get_fs(documents_directory)
    document_file_names = fs.ls(documents_directory)
    dgw = Spec2VecFSDataGateway()

    document_counter = 0

    for doc in FileSystemDocumentIterator(dgw, document_file_names):
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
