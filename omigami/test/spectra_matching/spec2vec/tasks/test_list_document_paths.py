from pathlib import Path

from omigami.spectra_matching.spec2vec.tasks.deploy_model_tasks import (
    ListDocumentPaths,
)
from omigami.spectra_matching.storage import FSDataGateway


def test_chunk_document_paths(tmpdir):
    paths = []
    for i in range(22):
        path = Path(str(tmpdir / f"doc-{i}"))
        path.touch()
        paths.append(str(path))

    fs_dgw = FSDataGateway()

    task = ListDocumentPaths(tmpdir, fs_dgw)
    document_paths = task.run()

    assert len(document_paths) == 22
    assert {str(p) for p in document_paths} == set(paths)
