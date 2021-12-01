from pathlib import Path

from omigami.spectra_matching.spec2vec.tasks.deploy_model_tasks import (
    ListDocumentPaths,
)
from omigami.spectra_matching.storage import FSDataGateway


def test_chunk_document_paths(tmpdir):
    _ = [Path(str(tmpdir / f"doc-{i}")).touch() for i in range(22)]
    fs_dgw = FSDataGateway()

    task = ListDocumentPaths(tmpdir, fs_dgw)
    document_paths = task.run()

    assert len(document_paths) == 22
