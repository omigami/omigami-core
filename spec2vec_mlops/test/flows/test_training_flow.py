import os
from typing import Union

import pytest
from prefect import Flow, unmapped
from prefect.engine.state import State

from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task
from spec2vec_mlops.tasks.train_model import train_model_task
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.load_spectrum_ids import load_spectrum_ids_task

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


def spec2vec_train_pipeline_local(
    source_uri: str,
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
        clean_data_task.map(raw)
        all_spectrum_ids_chunks = load_spectrum_ids_task(chunksize=1000)
        convert_to_documents_task.map(all_spectrum_ids_chunks, n_decimals=unmapped(2))
        model = train_model_task(iterations, window)
        run_id = register_model_task(
            mlflow_server_uri,
            model,
            experiment_name,
            save_model_path,
            n_decimals,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
        make_embeddings_task.map(
            unmapped(model),
            all_spectrum_ids_chunks,
            unmapped(run_id),
            unmapped(n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
    state = flow.run()
    return state


def test_spec2vec_train_pipeline_local(gnps_small_json, tmpdir):
    state = spec2vec_train_pipeline_local(
        source_uri=gnps_small_json,
        n_decimals=2,
        iterations=10,
        window=5,
        save_model_path=f"{tmpdir}/mflow",
        mlflow_server_uri=f"{tmpdir}/mlflow/",
        experiment_name="experiment",
    )
    assert state.is_successful()
