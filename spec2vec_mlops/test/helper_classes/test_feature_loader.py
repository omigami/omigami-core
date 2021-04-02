import os

import pytest
from spec2vec_mlops import config
from spec2vec_mlops.helper_classes.data_storer import SpectrumStorer, DocumentStorer
from spec2vec_mlops.helper_classes.feature_loader import FeatureLoader

FEAST_CORE_URL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


@pytest.fixture()
def feature_loader():
    return FeatureLoader(FEAST_CORE_URL)


@pytest.fixture()
def target_spectrum_ids(cleaned_data):
    return [cleaned_data[0].metadata["spectrum_id"]]


@pytest.fixture()
def spectrum_stored(tmpdir, cleaned_data):
    ds = SpectrumStorer(f"file://{tmpdir}", FEAST_CORE_URL)
    ds.store_cleaned_data(cleaned_data)


@pytest.fixture()
def documents_stored(tmpdir, documents_data):
    ds = DocumentStorer(f"file://{tmpdir}", FEAST_CORE_URL)
    ds.store_documents(documents_data)


def test_load_all_spectrum_ids(feature_loader, documents_stored):
    all_spectrum_ids = feature_loader.load_all_spectrum_ids()
    assert len(all_spectrum_ids) > 0


def test_load_cleaned_data(feature_loader, spectrum_stored, target_spectrum_ids):
    spectra = feature_loader.load_clean_data(target_spectrum_ids)
    assert len(spectra) > 0


def test_load_documents(feature_loader, documents_stored, target_spectrum_ids):
    documents = feature_loader.load_documents(target_spectrum_ids)
    assert len(documents) > 0
