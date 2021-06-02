from typing import Optional

from typing_extensions import Literal

from spec2vec_mlops.config import (
    SOURCE_URI_PARTIAL_GNPS,
    API_SERVER,
    PROJECT_NAME,
    MODEL_DIR,
    MLFLOW_SERVER,
    DATASET_DIR,
    RedisDBDatasetSize,
    S3_BUCKET,
)
from prefect import Client

from spec2vec_mlops.flows.training_flow import build_training_flow
from spec2vec_mlops.authentication.authenticator import KratosAuthenticator
from spec2vec_mlops.gateways.input_data_gateway import FSInputDataGateway

from spec2vec_mlops.gateways.redis_spectrum_gateway import (
    RedisSpectrumDataGateway,
)
from spec2vec_mlops.tasks.download_data import DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import ProcessSpectrumParameters
from spec2vec_mlops.tasks.train_model import TrainModelParameters

from spec2vec_mlops.flows.config import (
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
    dataset_name: str,
    n_decimals: int = 2,
    skip_if_exists: bool = True,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    output_dir: str = S3_BUCKET,
    project_name: str = PROJECT_NAME,
    model_output_dir: str = MODEL_DIR,
    mlflow_server: str = MLFLOW_SERVER,
    flow_name: str = "spec2vec-training-flow",
    environment: Literal["dev", "prod"] = "dev",
    deploy_model: bool = False,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):

    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(api_server=API_SERVER[environment], api_token=session_token)
    else:
        client = Client(api_server=API_SERVER[environment])
    client.create_project(project_name)

    # config values
    try:
        redis_db = RedisDBDatasetSize[dataset_name]
        dataset_path = DATASET_DIR[dataset_name] + "gnps.json"
        spectrum_ids_name = DATASET_DIR[dataset_name] + "spectrum_ids.pkl"
    except KeyError:
        raise ValueError(
            f"No such option available for reference dataset: {dataset_name}. Available options are:"
            f"{list(RedisDBDatasetSize.keys())}."
        )

    input_dgw = FSInputDataGateway()
    spectrum_dgw = RedisSpectrumDataGateway()

    download_parameters = DownloadParameters(
        source_uri, output_dir, dataset_path, input_dgw, spectrum_ids_name
    )
    process_parameters = ProcessSpectrumParameters(
        spectrum_dgw,
        input_dgw,
        n_decimals,
        skip_if_exists,
    )
    train_model_parameters = TrainModelParameters(spectrum_dgw, iterations, window)

    flow_config = make_flow_config(
        image=image,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=redis_db,
        environment=environment,
    )

    flow = build_training_flow(
        download_params=download_parameters,
        process_params=process_parameters,
        train_params=train_model_parameters,
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

    training_flow_id = client.register(
        flow,
        project_name=project_name,
    )
    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
    )
    return flow_run_id
