import time

from prefect import task

from mlops_presentation.model_and_registration import ModelRegister, SimpleModel


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
    run_id = model_register.register_model(
        SimpleModel(),
        experiment_name,
        path,
    )
    return run_id
