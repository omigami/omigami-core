import pickle
from pathlib import Path
import pytest
from spec2vec_mlops.helper_classes.data_loader import DataLoader

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


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


@pytest.fixture
def assets_dir():
    return Path(__file__).parents[0] / "assets"


@pytest.fixture()
def gnps_small_json(assets_dir):
    path = str(assets_dir / "SMALL_GNPS.json")
    return f"file://{path}"


@pytest.fixture()
def loaded_data(gnps_small_json):
    dl = DataLoader()
    return dl.load_gnps_json(gnps_small_json)


@pytest.fixture
def cleaned_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_cleaned.pickle")
    with open(path, "rb") as handle:
        cleaned_data = pickle.load(handle)
    return cleaned_data


@pytest.fixture
def documents_data(assets_dir):
    path = str(assets_dir / "SMALL_GNPS_as_documents.pickle")
    with open(path, "rb") as handle:
        documents_data = pickle.load(handle)
    return documents_data


@pytest.fixture
def word2vec_model(assets_dir):
    path = str(assets_dir / "model.pickle")
    with open(path, "rb") as handle:
        model = pickle.load(handle)
    return model


@pytest.fixture()
def embeddings(documents_data, word2vec_model):
    em = EmbeddingMaker()
    res = []
    for document in documents_data:
        res.append(
            em.make_embedding(
                model=word2vec_model,
                document=document,
                intensity_weighting_power=0.5,
                allowed_missing_percentage=5.0,
            )
        )
    return res
