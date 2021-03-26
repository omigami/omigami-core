import logging

import click
from prefect import Flow, Parameter, Client, unmapped
from prefect.engine.state import State
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.store_cleaned_data_and_words import (
    store_cleaned_data_task,
    store_words_task,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# variable definitions
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"].get(str)
API_SERVER_REMOTE = config["prefect_flow_registration"]["api_server"]["remote"].get(str)
API_SERVER_LOCAL = config["prefect_flow_registration"]["api_server"]["local"].get(str)
FEAST_CORE_URL_REMOTE = config["feast"]["url"]["remote"].get(str)


def spec2vec_train_pipeline_local(
    source_uri: str, feast_source_dir: str, feast_core_url: str
) -> State:
    with Flow("flow") as flow:
        raw = load_data_task(source_uri)
        logger.info("Data loading is complete.")
        cleaned = clean_data_task.map(raw)
        logger.info("Data cleaning is complete.")
        store_cleaned_data_task(cleaned, feast_source_dir, feast_core_url)
        documents = convert_to_documents_task.map(cleaned, n_decimals=unmapped(2))
        store_words_task(documents, feast_source_dir, feast_core_url)
    state = flow.run()
    return state


def spec2vec_train_pipeline_distributed(
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,  # TODO when running in prod set to SOURCE_URI_COMPLETE_GNPS
    api_server: str = API_SERVER_REMOTE,
    project_name: str = "spec2vec-mlops-project-documents-task",
    feast_source_dir: str = "s3://dr-prefect/spec2vec-training-flow/",
    feast_core_url: str = FEAST_CORE_URL_REMOTE,
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

    Returns
    -------
    flow_run_id: unique flow_run_id registered in Prefect

    """
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools/prefect:spec2vec_mlops-SNAPSHOT.63b1441",
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
        store_words_task(documents, feast_source_dir, feast_core_url)
        # encoded = encode_training_data_task(documents)
        # trained = train_model_task(documents)
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
