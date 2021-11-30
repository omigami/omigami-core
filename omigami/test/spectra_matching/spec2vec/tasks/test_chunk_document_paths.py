from pathlib import Path

from omigami.spectra_matching.spec2vec.tasks.deploy_model_tasks import (
    ChunkDocumentPaths,
)
from omigami.spectra_matching.storage import FSDataGateway


def test_chunk_document_paths(tmpdir):
    _ = [Path(str(tmpdir / f"doc-{i}")).touch() for i in range(22)]
    fs_dgw = FSDataGateway()

    task = ChunkDocumentPaths(tmpdir, fs_dgw, chunk_size=5)
    chunked_paths = task.run()

    assert len(chunked_paths) == 5
    assert len(chunked_paths[0]) == 5
    assert len(chunked_paths[-1]) == 2
