from datetime import datetime
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

# -- CONFIGS:

# - GNPS JSON
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri"]["complete"]
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri"]["partial"]

# - PREFECT
API_SERVER = config["prefect_flow_registration"]["api_server"]
PROJECT_NAME = config["prefect"]["project"]
OUTPUT_DIR = config["prefect"]["output_dir"]
DATASET_FOLDER = config["prefect"]["dataset_folder"]
DATASET_DIR = (
    f"{DATASET_FOLDER}/{datetime.now().date()}/"
)
MODEL_DIR = config["prefect"]["model_folder"]
S3_MODEL_BUCKET = config["prefect"]["s3_model_bucket"]
DATASET_NAME = DATASET_DIR + "gnps.json"
SPECTRUM_IDS_NAME = DATASET_DIR + "spectrum_ids.pkl"

# - MLFLOW
MLFLOW_SERVER = config["mlflow"]["url"]["remote"]


# TODO(bk): Should this be a global config? Shouldnt this be created close to where its used
#      and receive the necessary params?
FLOW_CONFIG = {
    "run_config": KubernetesRun(
        job_template_path=str(Path(__file__).parents[0] / "job_spec.yaml"),
        labels=["dev"],
        service_account_name="prefect-server-serviceaccount",
        env={"REDIS_HOST": "redis-master.redis", "REDIS_DB": "2"},
    ),
    "storage": S3(S3_MODEL_BUCKET),
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
    # TODO(bk): if those below are not passed via CLI, probably we dont need them as parameters as
    #       we can just use the global configs directly
    api_server: str = API_SERVER,
    dataset_name: str = DATASET_NAME,
    spectrum_ids_name: str = SPECTRUM_IDS_NAME,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    output_dir: str = OUTPUT_DIR,
    project_name: str = PROJECT_NAME,
    model_output_dir: str = MODEL_DIR,
    mlflow_server: str = MLFLOW_SERVER,
    redis_db: str = "2",
):
    # TODO(bk): Modifying global values, things getting mixed up, bad design
    FLOW_CONFIG["run_config"].image = image
    # TODO(bk): Is there a reason for redis_db being specified in FLOW_CONFIG above if we have this parameter
    #       that overrides it anyway?
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
