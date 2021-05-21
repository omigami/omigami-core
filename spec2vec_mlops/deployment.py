from datetime import datetime
from typing import Optional

from spec2vec_mlops import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER,
    PROJECT_NAME,
    OUTPUT_DIR,
    DATASET_FOLDER,
    MODEL_DIR,
    MLFLOW_SERVER,
)
from prefect import Client
from prefect.executors import LocalDaskExecutor, DaskExecutor
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.authentication.authenticator import KratosAuthenticator
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway

from spec2vec_mlops.gateways.redis_spectrum_gateway import (
    RedisSpectrumDataGateway,
    DEFAULT_REDIS_DB_ID,
    RedisDBDatasetSize,
)
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrumParameters

from spec2vec_mlops.flows.config import (
    make_flow_config,
    PrefectRunMethods,
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
    skip_if_exists: bool = True,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_server: str = API_SERVER,
    dataset: str = "small",
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    output_dir: str = OUTPUT_DIR,
    project_name: str = PROJECT_NAME,
    model_output_dir: str = MODEL_DIR,
    mlflow_server: str = MLFLOW_SERVER,
    redis_db: str = "2",
    flow_name: str = "spec2vec-training-flow",
    dataset_size: str = None,
    dataset_name: str = None,
    spectrum_ids_name: str = None,
):



    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(api_server=api_server, api_token=session_token)
    else:
        client = Client(api_server=api_server)
    client.create_project(project_name)

    # config values
    dataset_name = dataset_name or f"{DATASET_FOLDER}/{datetime.now().date()}/gnps.json"
    spectrum_ids_name = (
        spectrum_ids_name
        or f"{DATASET_FOLDER}/{datetime.now().date()}/spectrum_ids.pkl"
    )

    if dataset_size is not None:
        try:
            redis_db = RedisDBDatasetSize[dataset_size]
        except KeyError:
            raise ValueError(
                f"No such option available for reference dataset size: {dataset_size}."
            )
    else:
        redis_db = DEFAULT_REDIS_DB_ID

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

    # create flow config
    flow_config = make_flow_config(
        image=image,
        run_config_type=PrefectRunMethods.KUBERNETES,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=redis_db,
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
        flow_config=flow_config,
        redis_db=redis_db,
        flow_name=flow_name,
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
