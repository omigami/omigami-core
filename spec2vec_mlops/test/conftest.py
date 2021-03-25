from pathlib import Path

import pytest
from spec2vec_mlops.tasks.clean_data import DataCleaner
from spec2vec_mlops.tasks.convert_to_documents import DocumentConverter
from spec2vec_mlops.tasks.load_data import DataLoader


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
def cleaned_data(gnps_small_json):
    dl = DataLoader()
    dc = DataCleaner()

    loaded_data = dl.load_gnps_json(gnps_small_json)
    return [dc.clean_data(spectrum) for spectrum in loaded_data]


@pytest.fixture
def documents_data(cleaned_data):
    converter = DocumentConverter()
    return [converter.convert_to_document(spectrum, 1) for spectrum in cleaned_data]
