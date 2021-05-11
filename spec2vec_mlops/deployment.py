from pathlib import Path
from typing import Optional

from prefect import Client
from prefect.executors import LocalDaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.authentication.authenticator import KratosAuthenticator
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway

from spec2vec_mlops.gateways.redis_gateway import RedisSpectrumDataGateway
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrumParameters


# TODO: move all of these variables to config default and organize it
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]
API_SERVER = config["prefect_flow_registration"]["api_server"]
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]
PROJECT_NAME = "spec2vec-mlops-project-spec2vec-load-10k-data-pt-3"
OUTPUT_DIR = "s3://dr-prefect"
DATASET_NAME = "spec2vec-training-flow/downloaded_datasets/small/gnps.json"
SPECTRUM_IDS_NAME = "spec2vec-training-flow/downloaded_datasets/spectrum_ids"
MODEL_DIR = "s3://dr-prefect/spec2vec-training-flow/mlflow"
FLOW_CONFIG = {
    "run_config": KubernetesRun(
        job_template_path=str(Path(__file__).parents[0] / "job_spec.yaml"),
        labels=["dev"],
        service_account_name="prefect-server-serviceaccount",
        env={"REDIS_HOST": "redis-master.redis", "REDIS_DB": "2"},
    ),
    "storage": S3("dr-prefect"),
    # TODO: maybe also useful to have as a parameter?
    "executor": LocalDaskExecutor(scheduler="threads", num_workers=5),
    # "executor": DaskExecutor(address="dask-scheduler.dask:8786"),
}


def deploy_training_flow(
    image: str,
    iterations: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    n_decimals: int = 2,
    skip_if_exists: bool = True,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_server: str = API_SERVER,
    dataset_name: str = DATASET_NAME,
    spectrum_ids_name: str = SPECTRUM_IDS_NAME,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    output_dir: str = OUTPUT_DIR,
    project_name: str = PROJECT_NAME,
    model_output_dir: str = MODEL_DIR,
    mlflow_server: str = MLFLOW_SERVER,
    redis_db: int = 2,
):
    FLOW_CONFIG["run_config"].image = image
    FLOW_CONFIG["run_config"].env["REDIS_DB"] = redis_db
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(api_server=api_server, api_token=session_token)
    else:
        client = Client(api_server=api_server)
    client.create_project(project_name)

    input_dgw = FSInputDataGateway()
    spectrum_dgw = RedisSpectrumDataGateway()

    download_parameters = DownloadParameters(
        source_uri, output_dir, dataset_name, input_dgw, spectrum_ids_name
    )
    process_parameters = ProcessSpectrumParameters(
        spectrum_dgw,
        input_dgw,
        n_decimals,
        skip_if_exists,
    )

    flow = build_training_flow(
        download_params=download_parameters,
        process_params=process_parameters,
        iterations=iterations,
        window=window,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        project_name=project_name,
        model_output_dir=model_output_dir,
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
