from datetime import timedelta
from typing import Optional

from prefect import Client
from typing_extensions import Literal

from omigami.authentication.authenticator import KratosAuthenticator
from omigami.config import (
    API_SERVER_URLS,
    PROJECT_NAME,
    MLFLOW_SERVER,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.ms2deepscore.config import MODEL_DIRECTORIES
from omigami.ms2deepscore.flows.minimal_flow import (
    build_minimal_flow,
    MinimalFlowParameters,
)
from omigami.spec2vec.gateways.input_data_gateway import FSInputDataGateway


def deploy_minimal_flow(
    image: str,
    project_name: str = PROJECT_NAME,
    mlflow_server: str = MLFLOW_SERVER,
    flow_name: str = "ms2deepscore-minimal-flow",
    environment: Literal["dev", "prod"] = "dev",
    deploy_model: bool = False,
    overwrite: bool = False,
    schedule: timedelta = None,
    auth: bool = False,
    auth_url: Optional[str] = None,
    username: Optional[str] = None,
    password: Optional[str] = None,
    api_server: Optional[str] = None,
):
    """
    Deploys a model minimal flow to Prefect cloud and optionally deploys it as a
    Seldon deployment.

    1) authenticate user credentials using the auth service endpoint
    2) process database and filesystem configs
    3) instantiate gateways and task parameters
    4) create prefect flow configuration
    5) builds the minimal flow graph
    6) pushes the flow to prefect cloud

    """

    # authentication step
    api_server = api_server or API_SERVER_URLS[environment]
    if auth:
        authenticator = KratosAuthenticator(auth_url, username, password)
        session_token = authenticator.authenticate()
        client = Client(api_server=api_server, api_token=session_token)
    else:
        client = Client(api_server=API_SERVER_URLS[environment])
    client.create_project(project_name)

    # validates parameters and use them to get task configuration variables
    if environment not in ["dev", "prod"]:
        raise ValueError("Environment not valid. Should be either 'dev' or 'prod'.")

    mlflow_output_dir = MODEL_DIRECTORIES[environment]["mlflow"]

    input_dgw = FSInputDataGateway()

    flow_parameters = MinimalFlowParameters(
        input_dgw=input_dgw,
        overwrite=overwrite,
        environment=environment,
    )

    flow_config = make_flow_config(
        image=image,
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        environment=environment,
        schedule=schedule,
    )

    flow = build_minimal_flow(
        project_name,
        flow_name,
        flow_config,
        flow_parameters,
        mlflow_output_dir=mlflow_output_dir,
        mlflow_server=mlflow_server,
        deploy_model=deploy_model,
    )

    minimal_flow_id = client.register(
        flow,
        project_name=project_name,
    )

    flow_run_id = client.create_flow_run(
        flow_id=minimal_flow_id,
        run_name=f"run {project_name}",
    )

    return flow_run_id
