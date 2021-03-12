from pathlib import Path

import pytest

from tasks.load_data import DataLoader


@pytest.fixture()
def gnps_small_json():
    ASSET_DIR = str(Path(__file__).parents[1] / "assets" / "SMALL_GNPS.json")
    return f"file://{ASSET_DIR}"


@pytest.mark.skip(reason="this test requires internet connection.")
@pytest.mark.parametrize(
    "uri",
    [
        "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json",
        "https://gnps-external.ucsd.edu/gnpslibrary/ALL_GNPS.json",
    ],
)
def test_load_gnps_json_with_web_uri(uri):
    dl = DataLoader()

    for item in dl.load_gnps_json(uri):
        assert isinstance(item, dict)


def test_load_gnps_json_with_local_uri(gnps_small_json):
    dl = DataLoader()

    for item in dl.load_gnps_json(gnps_small_json):
        assert isinstance(item, dict)
        assert item["spectrum_id"]
