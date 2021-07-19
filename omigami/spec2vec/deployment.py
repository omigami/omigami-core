from datetime import timedelta, datetime
from typing import Optional

from prefect import Client
from typing_extensions import Literal

from omigami.authentication.authenticator import KratosAuthenticator
from omigami.config import (
    API_SERVER_URLS,
    PROJECT_NAME,
    MLFLOW_SERVER,
    REDIS_DATABASES,
    S3_BUCKETS,
    IonModes,
    DATASET_IDS,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.spec2vec.config import (
    MODEL_DIRECTORIES,
)
from omigami.spec2vec.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spectrum_cleaner import SpectrumCleaner


def deploy_training_flow(
    image: str,
    iterations: int,
    window: int,
    intensity_weighting_power: float,
    allowed_missing_percentage: float,
    dataset_name: str,
    n_decimals: int = 2,
    chunk_size: int = int(1e8),
    ion_mode: IonModes = "positive",
    skip_if_exists: bool = True,
    source_uri: str = SOURCE_URI_PARTIAL_GNPS,
    project_name: str = PROJECT_NAME,
    mlflow_server: str = MLFLOW_SERVER,
    flow_name: str = "spec2vec-training-flow",
    environment: Literal["dev", "prod"] = "dev",
    deploy_model: bool = False,
    overwrite: bool = False,
    schedule: timedelta = None,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
):
    """
    Deploys a model training flow to Prefect cloud and optionally deploys it as a Seldon deployment.

    1) authenticate user credentials using the auth service endpoint
    2) process database and filesystem configs
    3) instantiate gateways and task parameters
    4) create prefect flow configuration
    5) builds the training flow graph
    6) pushes the flow to prefect cloud

    """

    # authentication step
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(
            api_server=API_SERVER_URLS[environment], api_token=session_token
        )
    else:
        client = Client(api_server=API_SERVER_URLS[environment])
    client.create_project(project_name)

    # validates parameters and use them to get task configuration variables
    if environment not in ["dev", "prod"]:
        raise ValueError("Environment not valid. Should be either 'dev' or 'prod'.")

    if dataset_name not in DATASET_IDS[environment].keys():
        raise ValueError(
            f"No such option available for reference dataset: {dataset_name}. Available options are:"
            f"{list(DATASET_IDS[environment].keys())}."
        )

    model_output_dir = MODEL_DIRECTORIES[environment]
    dataset_id = DATASET_IDS[environment][dataset_name].format(date=datetime.today())
    redis_db = REDIS_DATABASES[environment][dataset_name]
    output_dir = S3_BUCKETS[environment]

    input_dgw = FSInputDataGateway()
    spectrum_dgw = Spec2VecRedisSpectrumDataGateway()
    cleaner = SpectrumCleaner()

    flow_parameters = TrainingFlowParameters(
        input_dgw=input_dgw,
        spectrum_dgw=spectrum_dgw,
        cleaner=cleaner,
        source_uri=source_uri,
        output_dir=output_dir,
        dataset_id=dataset_id,
        chunk_size=chunk_size,
        ion_mode=ion_mode,
        n_decimals=n_decimals,
        skip_if_exists=skip_if_exists,
        iterations=iterations,
        window=window,
        overwrite=overwrite,
    )

    flow_config = make_flow_config(
        image=image,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db=redis_db,
        environment=environment,
        schedule=schedule,
    )

    flow = build_training_flow(
        project_name,
        flow_name,
        flow_config,
        flow_parameters,
        intensity_weighting_power=intensity_weighting_power,
        allowed_missing_percentage=allowed_missing_percentage,
        model_output_dir=model_output_dir,
        mlflow_server=mlflow_server,
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
