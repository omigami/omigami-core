import os
from pathlib import Path
from typing import Union

import pytest
from drfs import DRPath
from prefect import Flow, unmapped, case
from prefect.engine.state import State

from spec2vec_mlops import config
from spec2vec_mlops.tasks.check_condition import check_condition, check_model_condition
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.download_data import download_data_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.train_model import train_model_task

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[3])


pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)


def spec2vec_train_pipeline_local(
    source_uri: str,
    download_out_dir: DRPath,
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
        file_path = download_data_task(source_uri, download_out_dir)
        raw_chunks = load_data_task(file_path, chunksize=5000)
        all_spectrum_ids_chunks = clean_data_task.map(
            raw_chunks, n_decimals=unmapped(2)
        )

        with case(check_condition(all_spectrum_ids_chunks), True):
            model_pos = train_model_task(iterations, window, "positive")
            with case(check_model_condition(model_pos), True):
                run_id_pos = register_model_task(
                    mlflow_server_uri,
                    model_pos,
                    experiment_name,
                    save_model_path,
                    n_decimals,
                    intensity_weighting_power,
                    allowed_missing_percentage,
                )
                make_embeddings_task.map(
                    unmapped(model_pos),
                    all_spectrum_ids_chunks,
                    unmapped(run_id_pos),
                    unmapped(n_decimals),
                    unmapped("positive"),
                    unmapped(intensity_weighting_power),
                    unmapped(allowed_missing_percentage),
                )

            model_neg = train_model_task(iterations, window, "negative")
            with case(check_model_condition(model_neg), True):
                run_id_neg = register_model_task(
                    mlflow_server_uri,
                    model_neg,
                    experiment_name,
                    save_model_path,
                    n_decimals,
                    intensity_weighting_power,
                    allowed_missing_percentage,
                )
                make_embeddings_task.map(
                    unmapped(model_neg),
                    all_spectrum_ids_chunks,
                    unmapped(run_id_neg),
                    unmapped(n_decimals),
                    unmapped("negative"),
                    unmapped(intensity_weighting_power),
                    unmapped(allowed_missing_percentage),
                )

    state = flow.run()
    return state


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
