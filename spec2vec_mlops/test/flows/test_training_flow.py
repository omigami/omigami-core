import os
from pathlib import Path
from typing import Union

import pytest
from prefect import Flow, unmapped
from prefect.engine.state import State

from spec2vec_mlops import config
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.store_cleaned_data import store_cleaned_data_task
from spec2vec_mlops.tasks.store_documents import store_documents_task
from spec2vec_mlops.tasks.train_model import train_model_task

FEAST_CORE_URL_LOCAL = config["feast"]["url"]["local"].get(str)
os.chdir(Path(__file__).parents[3])


pytestmark = pytest.mark.skip(
    "This test can only be run if the Feast docker-compose is up"
)


def spec2vec_train_pipeline_local(
    source_uri: str,
    feast_source_dir: str,
    feast_core_url: str,
    n_decimals: int,
    save_model_path: str,
    mlflow_server_uri: str,
    experiment_name: str,
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> State:
    with Flow("flow") as flow:
        raw = load_data_task(source_uri)
        cleaned = clean_data_task.map(raw)
        store_cleaned_data_task(cleaned, feast_source_dir, feast_core_url)
        documents = convert_to_documents_task.map(
            cleaned, n_decimals=unmapped(n_decimals)
        )
        store_documents_task(documents, feast_source_dir, feast_core_url)
        model = train_model_task(documents, iterations, window)
        register_model_task(
            mlflow_server_uri,
            model,
            experiment_name,
            save_model_path,
            n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
        embeddings = make_embeddings_task.map(
            unmapped(model),
            documents,
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
    state = flow.run()
    return state


def test_spec2vec_train_pipeline_local(gnps_small_json, tmpdir):
    state = spec2vec_train_pipeline_local(
        source_uri=gnps_small_json,
        feast_source_dir=f"file://{tmpdir}",
        feast_core_url=FEAST_CORE_URL_LOCAL,
        n_decimals=2,
        iterations=10,
        window=5,
        save_model_path=f"{tmpdir}/mflow",
        mlflow_server_uri=f"{tmpdir}/mlflow/",
        experiment_name="experiment",
    )
    assert state.is_successful()
