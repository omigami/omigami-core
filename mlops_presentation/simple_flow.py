import logging

import click
from prefect import Flow, Client
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from mlops_presentation.tasks import (
    load_data_task,
    clean_data_task,
    train_model_task,
    register_model_task,
)
from spec2vec_mlops import config
from spec2vec_mlops.tasks.deploy_model import deploy_model_task
from spec2vec_mlops.utility.authenticator import KratosAuthenticator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# variable definitions
API_SERVER_REMOTE = config["prefect_flow_registration"]["api_server"]["remote"]
API_SERVER_LOCAL = config["prefect_flow_registration"]["api_server"]["local"]
MLFLOW_SERVER_REMOTE = config["mlflow"]["url"]["remote"]


def simple_flow(
    api_server: str = API_SERVER_REMOTE,
    project_name: str = "simple-flow",
    save_model_path: str = "s3://dr-prefect/spec2vec-training-flow/mlflow",
    mlflow_server_uri: str = MLFLOW_SERVER_REMOTE,
    seldon_deployment_path: str = "spec2vec_mlops/seldon_deployment.yaml",
    session_token: str = None,
) -> str:
    """Function to register Prefect flow using remote cluster
    Returns
    -------
    flow_run_id: unique flow_run_id registered in Prefect

    """
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools/prefect:spec2vec_mlops-SNAPSHOT.4290f75",
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
        ),
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(),
    }
    with Flow("simple-training-flow", **custom_confs) as training_flow:
        data = load_data_task()
        cleaned_data = clean_data_task(data)
        model = train_model_task(cleaned_data)
        run_id = register_model_task(
            mlflow_server_uri,
            model,
            project_name,
            save_model_path,
        )
        deploy_model_task(run_id, seldon_deployment_path)

    if session_token:
        client = Client(api_server=api_server, api_token=session_token)
    else:
        client = Client(api_server=api_server)
    client.create_project(project_name)
    training_flow_id = client.register(
        training_flow,
        project_name=project_name,
    )
    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
    )
    return flow_run_id


@click.group()
def cli():
    pass


@cli.command(name="register-train-pipeline")
@click.option("--auth", default=False, help="Enable authentication")
@click.option("--auth_url", default=None, help="Kratos Public URI")
@click.option("--username", default=None, help="Login username")
@click.option("--password", default=None, help="Login password")
def register_simple_flow_cli(auth, auth_url, username, password, *args, **kwargs):
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        kwargs["session_token"] = session_token
    simple_flow(*args, **kwargs)


if __name__ == "__main__":
    cli()
