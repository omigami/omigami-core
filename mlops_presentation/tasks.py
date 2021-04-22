import time

from prefect import task

from mlops_presentation.simple_model import SimpleModel
from spec2vec_mlops.helper_classes.model_register import ModelRegister


@task()
def load_data_task():
    time.sleep(3)
    return True


@task()
def clean_data_task(data):
    time.sleep(4)
    return True


@task()
def train_model_task(data):
    time.sleep(5)
    return True


@task()
def register_model_task(
    server_uri: str,
    model,
    experiment_name: str,
    path: str,
) -> str:
    model_register = ModelRegister(server_uri)
    params = {
        "parameter1": 100,
        "parameter2": 5,
        "parameter3": 1,
    }
    metrics = {
        "metric1": 1,
        "metric2": 80,
        "metric3": 6,
    }
    run_id = model_register.register_model(
        SimpleModel(),
        params=params,
        metrics=metrics,
        experiment_name=experiment_name,
        path=path,
        code_to_save=["mlops_presentation"],
    )
    return run_id
