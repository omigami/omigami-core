import os

import pytest

from omigami.spectra_matching.spec2vec.storage.fs_document_iterator import (
    FileSystemDocumentIterator,
)
from omigami.spectra_matching.storage import FSDataGateway


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_read_documents_iter(documents_stored, s3_documents_directory, s3_mock):

    fs_dgw = FSDataGateway()
    document_file_names = fs_dgw.list_files(s3_documents_directory)
    iterator = FileSystemDocumentIterator(fs_dgw, document_file_names)

    document_counter = 0

    for doc in iterator:
        document_counter += 1

    assert document_counter == len(document_file_names) * 10
