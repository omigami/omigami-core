from typing import Optional

import click
from prefect import Client
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.authentication.authenticator import KratosAuthenticator
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway
from spec2vec_mlops.tasks.utils import add_options

SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
API_SERVER = config["prefect_flow_registration"]["api_server"]
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]
PROJECT_NAME = "spec2vec-mlops-project-spec2vec-load-10k-data-pt-3"
DATASET_DIR = "s3://dr-prefect/spec2vec-training-flow/downloaded_datasets/"
DATASET_ID = "test_10k"  # we also have 'small'
MODEL_DIR = "s3://dr-prefect/spec2vec-training-flow/mlflow"
SELDON_DEPLOYMENT_PATH = "spec2vec_mlops/seldon_deployment.yaml"
FLOW_CONFIG = {
    "run_config": KubernetesRun(
        image="drtools/prefect:spec2vec_mlops-SNAPSHOT.3c7cf0c",
        job_template_path="./spec2vec_mlops/job_spec.yaml",
        labels=["dev"],
        service_account_name="prefect-server-serviceaccount",
        env={"REDIS_HOST": "redis-master.redis", "REDIS_DB": "2"},
    ),
    "storage": S3("dr-prefect"),
    # TODO: maybe also useful to have as a parameter?
    "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    # "executor": DaskExecutor(address="dask-scheduler.dask:8786"),
}


# I'm not sure we will ever want to expose this on the CLI so I am not including these
# parameters right now. If you think we won't need this we can just tweak config_default
# for changes in configuration. Atm it doesn't look these change often
configuration_options = [
    click.option("--project-name", default=PROJECT_NAME),
    click.option("--source-uri", default=SOURCE_URI_PARTIAL_GNPS),
    click.option("--dataset-dir", default=DATASET_DIR),
    click.option("--model-output-dir", default=MODEL_DIR),
    click.option("--seldon-deployment-path", default=SELDON_DEPLOYMENT_PATH),
    click.option("--mlflow-server", default=MLFLOW_SERVER["remote"]),
]

auth_options = [
    click.option(
        "--api-server", default=API_SERVER["remote"], help="URL to the prefect API"
    ),
    click.option("--auth", default=False, help="Enable authentication"),
    click.option("--auth_url", default=None, help="Kratos Public URI"),
    click.option("--username", default=None, help="Login username"),
    click.option("--password", default=None, help="Login password"),
]


@click.group()
def cli():
    pass


@cli.command(name="register-training-flow")
@click.option("--dataset-id", default=DATASET_ID)
@click.option("--n-decimals", type=int, default=2)
@click.option("--iterations", type=int, default=25)
@click.option("--window", type=int, default=500)
@click.option("--intensity-weighting-power", type=float, default=0.5)
@click.option("--allowed-missing-percentage", type=float, default=5.0)
@add_options(auth_options)
def deploy_training_flow(
    dataset_name: str,
    n_decimals: int,
    iterations: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    api_server: str,
    auth: bool,
    auth_url: Optional[str],
    username: Optional[str],
    password: Optional[str],
    project_name: str = PROJECT_NAME,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    dataset_dir: str = DATASET_DIR,
    model_output_dir: str = MODEL_DIR,
    seldon_deployment_path: str = SELDON_DEPLOYMENT_PATH,
    mlflow_server: str = MLFLOW_SERVER["remote"],
):
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(api_server=api_server, api_token=session_token)
    else:
        client = Client(api_server=api_server)
    client.create_project(project_name)
    input_dgw = FSInputDataGateway()

    flow = build_training_flow(
        dataset_name=dataset_name,
        n_decimals=n_decimals,
        iterations=iterations,
        window=window,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        project_name=project_name,
        input_dgw=input_dgw,
        redis_dgw=None,
        input_uri=source_uri,
        output_dir=dataset_dir,
        model_output_dir=model_output_dir,
        seldon_deployment_path=seldon_deployment_path,
        mlflow_server=mlflow_server,
        flow_config=FLOW_CONFIG,
    )

    training_flow_id = client.register(
        flow,
        project_name=project_name,
    )
    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
    )
    return flow_run_id


@cli.command(name="register-all-flows")
def deploy_model_cli():
    # spec2vec_model_deployment_pipeline_distributed()
    # flow = build_training_flow()
    pass


if __name__ == "__main__":
    cli()
