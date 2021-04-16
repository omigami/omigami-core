import logging
from typing import Union

import click
from drfs import DRPath
from prefect import Flow, Parameter, Client, unmapped, case
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.tasks.check_condition import check_condition
from spec2vec_mlops.tasks.clean_data import clean_data_task
from spec2vec_mlops.tasks.convert_to_documents import convert_to_documents_task
from spec2vec_mlops.tasks.deploy_model import deploy_model_task
from spec2vec_mlops.tasks.download_data import download_data_task
from spec2vec_mlops.tasks.load_data import load_data_task
from spec2vec_mlops.tasks.load_spectrum_ids import load_spectrum_ids_task
from spec2vec_mlops.tasks.make_embeddings import make_embeddings_task
from spec2vec_mlops.tasks.register_model import register_model_task
from spec2vec_mlops.tasks.train_model import train_model_task
from spec2vec_mlops.utility.authenticator import KratosAuthenticator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# variable definitions
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
API_SERVER_REMOTE = config["prefect_flow_registration"]["api_server"]["remote"]
API_SERVER_LOCAL = config["prefect_flow_registration"]["api_server"]["local"]
FEAST_CORE_URL_REMOTE = config["feast"]["url"]["remote"]
FEAST_SERVING_URL_REMOTE = config["feast"]["serving_url"]["remote"]
MLFLOW_SERVER_REMOTE = config["mlflow"]["url"]["remote"]


def spec2vec_train_pipeline_distributed(
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,  # TODO when running in prod set to SOURCE_URI_COMPLETE_GNPS
    api_server: str = API_SERVER_REMOTE,
    project_name: str = "spec2vec-mlops-project-spec2vec-load-small-data-5",
    download_out_dir: str = "s3://dr-prefect/spec2vec-training-flow/downloaded_datasets/small",  # or full if using complete GNPS
    n_decimals: int = 2,
    save_model_path: str = "s3://dr-prefect/spec2vec-training-flow/mlflow",
    mlflow_server_uri: str = MLFLOW_SERVER_REMOTE,
    conda_env_path: str = "requirements/environment.frozen.yaml",
    iterations: int = 25,
    window: int = 500,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    seldon_deployment_path: str = "spec2vec_mlops/seldon_deployment.yaml",
    session_token: str = None,
) -> str:
    """Function to register Prefect flow using remote cluster

    Parameters
    ----------
    source_uri: uri to load data from
    api_server: api_server to instantiate Client object
        when set to API_SERVER_LOCAL port-forwarding is required.
    project_name: name to register project in Prefect
    download_out_dir: location to save the downloaded datasets
    n_decimals: peak positions are converted to strings with n_decimal decimals
    save_model_path: path to save the trained model with MLFlow to
    mlflow_server_uri: url of MLFlow server
    conda_env_path: path to the conda environment requirements
    iterations: number of training iterations.
    window: window size for context words
    intensity_weighting_power: exponent used to scale intensity weights for each word
    allowed_missing_percentage: number of what percentage of a spectrum is allowed
        to be unknown to the model
    seldon_deployment_path: path to the seldon deployment configuration file

    Returns
    -------
    flow_run_id: unique flow_run_id registered in Prefect

    """
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools/prefect:spec2vec_mlops-SNAPSHOT.eec328f",
            labels=["dev"],
            service_account_name="prefect-server-serviceaccount",
            env={
                "FEAST_BASE_SOURCE_LOCATION": "s3a://dr-prefect/spec2vec-training-flow/feast",
                "FEAST_CORE_URL": FEAST_CORE_URL_REMOTE,
                "FEAST_SERVING_URL": FEAST_SERVING_URL_REMOTE,
                "FEAST_SPARK_LAUNCHER": "k8s",
                "FEAST_SPARK_K8S_NAMESPACE": "feast",
                "FEAST_SPARK_STAGING_LOCATION": "s3a://dr-prefect/spec2vec-training-flow/feast/staging",
                "FEAST_HISTORICAL_FEATURE_OUTPUT_FORMAT": "parquet",
                "FEAST_HISTORICAL_FEATURE_OUTPUT_LOCATION": "s3a://dr-prefect/spec2vec-training-flow/feast/output.parquet",
                "FEAST_HISTORICAL_FEATURE_OUTPUT_READ_LOCATION": "s3://dr-prefect/spec2vec-training-flow/feast/output.parquet",
                "FEAST_REDIS_HOST": "feast-redis-master.feast",
            },
        ),
        "storage": S3("dr-prefect"),
        "executor": LocalDaskExecutor(),
    }
    with Flow("spec2vec-training-flow", **custom_confs) as training_flow:
        uri = Parameter(name="uri")
        file_path = download_data_task(uri, DRPath(download_out_dir))
        raw_chunks = load_data_task(file_path, chunksize=1000)
        logger.info("Data loading is complete.")

        spectrum_ids_saved = clean_data_task.map(raw_chunks)
        logger.info("Data cleaning is complete.")

        with case(check_condition(spectrum_ids_saved), True):
            all_spectrum_ids_chunks = load_spectrum_ids_task(chunksize=1000)
            all_spectrum_ids_chunks = convert_to_documents_task.map(
                all_spectrum_ids_chunks, n_decimals=unmapped(2)
            )
            logger.info("Document conversion is complete.")

        with case(check_condition(all_spectrum_ids_chunks), True):
            model = train_model_task(iterations, window)
            run_id = register_model_task(
                mlflow_server_uri,
                model,
                project_name,
                save_model_path,
                n_decimals,
                intensity_weighting_power,
                allowed_missing_percentage,
                conda_env_path,
            )
            logger.info("Model training is complete.")

        make_embeddings_task.map(
            unmapped(model),
            all_spectrum_ids_chunks,
            unmapped(run_id),
            unmapped(n_decimals),
            unmapped(intensity_weighting_power),
            unmapped(allowed_missing_percentage),
        )
        logger.info("Saving embedding is complete.")
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
        parameters={"uri": source_uri},
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
def register_train_pipeline_cli(auth, auth_url, username, password, *args, **kwargs):
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        kwargs["session_token"] = session_token
    spec2vec_train_pipeline_distributed(*args, **kwargs)


@cli.command(name="register-all-flows")
def deploy_model_cli():
    # spec2vec_model_deployment_pipeline_distributed()
    spec2vec_train_pipeline_distributed()


if __name__ == "__main__":
    cli()
