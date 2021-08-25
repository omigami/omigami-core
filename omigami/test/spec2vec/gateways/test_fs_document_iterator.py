import pytest
import os

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.fs_document_iterator import FileSystemDocumentIterator


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_read_documents_iter(documents_stored, s3_documents_directory, s3_mock):

    fs_dgw = FSDataGateway()
    document_file_names = fs_dgw.listdir(s3_documents_directory)
    iterator = FileSystemDocumentIterator(fs_dgw, document_file_names)

    document_counter = 0

    for doc in iterator:
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
