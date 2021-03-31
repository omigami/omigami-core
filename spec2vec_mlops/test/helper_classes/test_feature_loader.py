import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import DataStorer
from spec2vec_mlops.helper_classes.feature_loader import FeatureLoader

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)


@pytest.fixture()
def feature_loader():
    return FeatureLoader(FEAST_CORE_URL)


@pytest.fixture()
def documents_stored(tmpdir, documents_data):
    ds = DataStorer(f"file://{tmpdir}", FEAST_CORE_URL)
    ds.store_documents(documents_data)


# @pytest.mark.skip("It can only be run if the Feast docker-compose is up")
def test_load_documents(feature_loader, documents_data):
    documents_df = feature_loader.load_documents()
    assert len(documents_df.columns) > 2
