from datetime import timedelta, datetime
from typing import Optional

from typing_extensions import Literal

from omigami.config import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER_URLS,
    PROJECT_NAME,
    MODEL_DIRECTORIES,
    MLFLOW_SERVER,
    DATASET_DIRECTORIES,
    REDIS_DATABASES,
    S3_BUCKETS,
)
from prefect import Client

from omigami.flows.training_flow import build_training_flow
from omigami.authentication.authenticator import KratosAuthenticator
from omigami.gateways.input_data_gateway import FSInputDataGateway

from omigami.gateways.redis_spectrum_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.tasks import (
    DownloadParameters,
    ProcessSpectrumParameters,
    TrainModelParameters,
)

from omigami.flows.config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)


def deploy_training_flow(
    image: str,
    iterations: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    n_decimals: int = 2,
    chunk_size: int = 1000,
    skip_if_exists: bool = True,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    project_name: str = PROJECT_NAME,
    mlflow_server: str = MLFLOW_SERVER,
    flow_name: str = "spec2vec-training-flow",
    environment: Literal["dev", "prod"] = "dev",
    deploy_model: bool = False,
    schedule: timedelta = None,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    dataset_name: Optional[str] = None,
):

    # authenticate user credentials using the auth service endpoint
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(
            api_server=API_SERVER_URLS[environment], api_token=session_token
        )
    else:
        client = Client(api_server=API_SERVER_URLS[environment])
    client.create_project(project_name)

    if environment not in ["dev", "prod"]:
        raise ValueError("Environment not valid. Should be either 'dev' or 'prod'.")

    # process database and filesystem configs
    if dataset_name not in DATASET_DIRECTORIES[environment].keys():
        raise ValueError(
            f"No such option available for reference dataset: {dataset_name}. Available options are:"
            f"{list(DATASET_DIRECTORIES[environment].keys())}."
        )

    model_output_dir = MODEL_DIRECTORIES[environment]
    dataset_folder = f'{DATASET_DIRECTORIES[environment][dataset_name]}/{datetime.today().strftime("%Y-%m")}'
    redis_db = REDIS_DATABASES[environment][dataset_name]
    output_dir = S3_BUCKETS[environment]

    # instantiate gateways
    input_dgw = FSInputDataGateway()
    spectrum_dgw = RedisSpectrumDataGateway()

    # instantiate required task parameters
    download_parameters = DownloadParameters(source_uri, output_dir, dataset_folder)
    process_parameters = ProcessSpectrumParameters(
        spectrum_dgw,
        n_decimals,
        skip_if_exists,
    )
    train_model_parameters = TrainModelParameters(spectrum_dgw, iterations, window)

    # create prefect flow configuration
    flow_config = make_flow_config(
        image=image,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=redis_db,
        environment=environment,
        schedule=schedule,
    )

    # executes the training flow procedure
    flow = build_training_flow(
        input_dgw=input_dgw,
        download_params=download_parameters,
        process_params=process_parameters,
        train_params=train_model_parameters,
        chunk_size=chunk_size,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        project_name=project_name,
        model_output_dir=model_output_dir,
        mlflow_server=mlflow_server,
        flow_config=flow_config,
        redis_db=redis_db,
        flow_name=flow_name,
        deploy_model=deploy_model,
    )

    # push the flow to prefect cloud
    training_flow_id = client.register(
        flow,
        project_name=project_name,
    )

    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
    )

    return flow_run_id
