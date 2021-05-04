import os
from pathlib import Path
from typing import Union

import pytest
from prefect import Flow, unmapped, case
from prefect.engine.state import State

from spec2vec_mlops import config
from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks.check_condition import check_condition
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.download_data import download_data
from spec2vec_mlops.tasks.load_data import load_data
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.train_model import train_model_task

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[3])


def spec2vec_train_pipeline_local(
    source_uri: str,
    download_out_dir: str,
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
        file_path = download_data(source_uri, FSInputDataGateway(), download_out_dir)
        raw_chunks = load_data(file_path, chunksize=5000)
        all_spectrum_ids_chunks = clean_data_task.map(
            raw_chunks, n_decimals=unmapped(2)
        )

        with case(check_condition(all_spectrum_ids_chunks), True):
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
        all_spectrum_ids_chunks = make_embeddings_task.map(
            unmapped(model),
            all_spectrum_ids_chunks,
            unmapped(run_id),
            unmapped(n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
    state = flow.run()
    return state


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_spec2vec_train_pipeline_local(tmpdir):
    state = spec2vec_train_pipeline_local(
        source_uri=SOURCE_URI_PARTIAL_GNPS,
        download_out_dir=tmpdir,
        n_decimals=2,
        iterations=10,
        window=5,
        save_model_path=f"{tmpdir}/mflow",
        mlflow_server_uri=f"{tmpdir}/mlflow/",
        experiment_name="experiment",
    )
    assert state.is_successful()


def test_training_flow():
    flow = build_training_flow(
        project_name="test",
        source_uri="source_uri",
        dataset_dir="datasets",
        dataset_id="dataset-id",
        model_output_dir="model-output",
        seldon_deployment_path="seldon-path",
        n_decimals=2,
        mlflow_server="mlflow-server",
        iterations=25,
        window=500,
        intensity_weighting_power=0.5,
        allowed_missing_percentage=5,
        flow_config=None,
    )

    assert flow
