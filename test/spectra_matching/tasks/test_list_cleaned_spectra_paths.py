from pathlib import Path

from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks.list_cleaned_spectra_paths import ListCleanedSpectraPaths


def test_list_cleaned_spectra_path(tmpdir):
    expected_paths = []
    for i in range(10):
        path = Path(str(tmpdir / f"cleaned_spectra-{i}"))
        path.touch()
        expected_paths.append(str(path))

    fs_dgw = FSDataGateway()

    task = ListCleanedSpectraPaths(tmpdir, fs_dgw)
    cleaned_spectra_paths = task.run()

    assert len(cleaned_spectra_paths) == 10
    assert {str(p) for p in cleaned_spectra_paths} == set(expected_paths)
