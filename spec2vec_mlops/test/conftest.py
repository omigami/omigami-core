import pickle
from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--longrun",
        action="store_true",
        dest="longrun",
        default=False,
        help="enable longrun-decorated tests",
    )


def pytest_configure(config):
    if not config.option.longrun:
        setattr(config.option, "markexpr", "not longrun")


@pytest.fixture()
def gnps_small_json():
    ASSET_DIR = str(Path(__file__).parents[0] / "assets" / "SMALL_GNPS.json")
    return f"file://{ASSET_DIR}"


@pytest.fixture
def cleaned_data():
    path = str(Path(__file__).parents[0] / "assets" / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture
def documents_data(cleaned_data):
    path = str(Path(__file__).parents[0] / "assets" / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documets_data = pickle.load(handle)
    return documets_data
