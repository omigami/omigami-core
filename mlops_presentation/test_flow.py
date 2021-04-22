import os
from pathlib import Path

from prefect import Flow
from prefect.engine.state import State

from mlops_presentation.tasks import (
    load_data_task,
    clean_data_task,
    train_model_task,
    register_model_task,
)
from spec2vec_mlops import config

SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
os.chdir(Path(__file__).parents[1])


def simple_flow_local(
    project_name: str,
    save_model_path: str,
    mlflow_server_uri: str,
) -> State:
    with Flow("flow") as flow:
        data = load_data_task()
        cleaned_data = clean_data_task(data)
        model = train_model_task(cleaned_data)
        run_id = register_model_task(
            mlflow_server_uri,
            model,
            project_name,
            save_model_path,
        )
    state = flow.run()
    return state


def test_simple_flow_local(tmpdir):
    state = simple_flow_local(
        project_name="experiment",
        save_model_path=f"{tmpdir}/mflow",
        mlflow_server_uri=f"{tmpdir}/mlflow/",
    )
    assert state.is_successful()
