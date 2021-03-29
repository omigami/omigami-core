import pytest
from prefect import Flow, unmapped
from prefect.engine.state import State

from spec2vec_mlops import config
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.store_cleaned_data import store_cleaned_data_task
from spec2vec_mlops.tasks.store_documents import store_documents_task
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task


FEAST_CORE_URL_LOCAL = config["feast"]["url"]["local"].get(str)

pytestmark = pytest.mark.skip(
    "This test can only be run if the Feast docker-compose is up"
)


def spec2vec_train_pipeline_local(
    source_uri: str, feast_source_dir: str, feast_core_url: str
) -> State:
    with Flow("flow") as flow:
        raw = load_data_task(source_uri)
        cleaned = clean_data_task.map(raw)
        store_cleaned_data_task(cleaned, feast_source_dir, feast_core_url)
        documents = convert_to_documents_task.map(cleaned, n_decimals=unmapped(2))
        store_documents_task(documents, feast_source_dir, feast_core_url)
    state = flow.run()
    return state


def test_spec2vec_train_pipeline_local(gnps_small_json, tmpdir):
    state = spec2vec_train_pipeline_local(
        source_uri=gnps_small_json,
        feast_source_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL_LOCAL,
    )
    assert state.is_successful()
