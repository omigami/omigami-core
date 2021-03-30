import logging
from typing import Union

import click
from prefect import Flow, Parameter, Client, unmapped
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.store_cleaned_data import store_cleaned_data_task
from spec2vec_mlops.tasks.store_documents import store_documents_task
from spec2vec_mlops.tasks.train_model import train_model_task
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# variable definitions
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"].get(str)
API_SERVER_REMOTE = config["prefect_flow_registration"]["api_server"]["remote"].get(str)
API_SERVER_LOCAL = config["prefect_flow_registration"]["api_server"]["local"].get(str)
FEAST_CORE_URL_REMOTE = config["feast"]["url"]["remote"].get(str)
MLFLOW_SERVER_REMOTE = config["mlflow"]["url"]["remote"].get(str)


def spec2vec_train_pipeline_distributed(
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,  # TODO when running in prod set to SOURCE_URI_COMPLETE_GNPS
    api_server: str = API_SERVER_REMOTE,
    project_name: str = "spec2vec-mlops-project-register-model-2",
    feast_source_dir: str = "s3://dr-prefect/spec2vec-training-flow/feast",
    feast_core_url: str = FEAST_CORE_URL_REMOTE,
    n_decimals: int = 2,
    save_model_path: str = "s3://dr-prefect/spec2vec-training-flow/mlflow",
    mlflow_server_uri: str = MLFLOW_SERVER_REMOTE,
    conda_env_path: str = "requirements/environment.frozen.yaml",
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> str:
    """Function to register Prefect flow using remote cluster

    Parameters
    ----------
    source_uri: uri to load data from
    api_server: api_server to instantiate Client object
        when set to API_SERVER_LOCAL port-forwarding is required.
    project_name: name to register project in Prefect
    feast_source_dir: location to save the file source of Feast
    feast_core_url: url where to connect to Feast server
    n_decimals: peak positions are converted to strings with n_decimal decimals
    save_model_path: path to save the trained model with MLFlow to
    iterations: number of training iterations.
    window: window size for context words
    intensity_weighting_power: exponent used to scale intensity weights for each word
    allowed_missing_percentage: number of what percentage of a spectrum is allowed
        to be unknown to the model

    Returns
    -------
    flow_run_id: unique flow_run_id registered in Prefect

    """
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools/prefect:spec2vec_mlops-SNAPSHOT.9a7b40a",
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
        ),
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(),
    }
    with Flow("spec2vec-training-flow", **custom_confs) as training_flow:
        uri = Parameter(name="uri")
        raw = load_data_task(uri)
        logger.info("Data loading is complete.")
        cleaned = clean_data_task.map(raw)
        logger.info("Data cleaning is complete.")
        store_cleaned_data_task(cleaned, feast_source_dir, feast_core_url)
        documents = convert_to_documents_task.map(cleaned, n_decimals=unmapped(2))
        store_documents_task(documents, feast_source_dir, feast_core_url)
        model = train_model_task(documents, iterations, window)
        register_model_task(
            mlflow_server_uri,
            model,
            project_name,
            save_model_path,
            n_decimals,
            conda_env_path,
        )
        embeddings = make_embeddings_task.map(
            unmapped(model),
            documents,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
    client = Client(api_server=api_server)
    client.create_project(project_name)
    training_flow_id = client.register(
        training_flow,
        project_name=project_name,
    )
    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
        parameters={"uri": source_uri},
    )
    return flow_run_id


@click.group()
def cli():
    pass


@cli.command(name="register-train-pipeline")
def register_train_pipeline_cli(*args, **kwargs):
    spec2vec_train_pipeline_distributed(*args, **kwargs)


@cli.command(name="register-all-flows")
def deploy_model_cli():
    # spec2vec_model_deployment_pipeline_distributed()
    spec2vec_train_pipeline_distributed()


if __name__ == "__main__":
    cli()
