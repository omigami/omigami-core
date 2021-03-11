import pytest

from tasks.load_data import DataLoader


@pytest.mark.skip(reason="this test requires internet connection.")
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_load_gnps_json_with_small_and_big_file(uri):
    dl = DataLoader()

    for item in dl.load_gnps_json(uri):
        assert isinstance(item, dict)
