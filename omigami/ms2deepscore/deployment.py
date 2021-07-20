from datetime import timedelta, datetime
from typing import Optional

from prefect import Client
from typing_extensions import Literal

from omigami.authentication.authenticator import KratosAuthenticator
from omigami.config import (
    API_SERVER_URLS,
    PROJECT_NAME,
    MLFLOW_SERVER,
    DATASET_IDS,
    REDIS_DATABASES,
    IonModes,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.config import MODEL_DIRECTORIES
from omigami.ms2deepscore.flows.minimal_flow import (
    build_minimal_flow,
    MinimalFlowParameters,
)
from omigami.ms2deepscore.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


class Deployer:
    def __init__(
        self,
        image: str,
        dataset_name: str,
        project_name: str = PROJECT_NAME,
        mlflow_server: str = MLFLOW_SERVER,
        environment: Literal["dev", "prod"] = "dev",
        ion_mode: IonModes = "positive",
        deploy_model: bool = False,
        overwrite: bool = False,
        schedule: timedelta = None,
        auth: bool = False,
        auth_url: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        api_server: Optional[str] = None,
        overwrite_all: bool = True,
        n_chunks: int = 10,
    ):
        self.image = image
        self.project_name = project_name
        self.mlflow_server = mlflow_server
        self.ion_mode = ion_mode
        self.deploy_model = deploy_model
        self.overwrite = overwrite
        self.schedule = schedule
        self.auth = auth
        self.auth_url = auth_url
        self.username = username
        self.password = password
        self.api_server = api_server
        self.overwrite_all = overwrite_all
        self.n_chunks = n_chunks

        if environment not in ["dev", "prod"]:
            raise ValueError("Environment not valid. Should be either 'dev' or 'prod'.")

        if dataset_name not in DATASET_IDS[environment].keys():
            raise ValueError(
                f"No such option available for reference dataset: {dataset_name}. Available options are:"
                f"{list(DATASET_IDS[environment].keys())}."
            )

        self.environment = environment
        self.dataset_name = dataset_name
        self.redis_db = REDIS_DATABASES[environment][dataset_name]

        self.input_dgw = FSInputDataGateway()
        self.spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    def _authenticate(self):
        api_server = self.api_server or API_SERVER_URLS[self.environment]
        if self.auth:
            authenticator = KratosAuthenticator(
                self.auth_url, self.username, self.password
            )
            session_token = authenticator.authenticate()
            client = Client(api_server=api_server, api_token=session_token)
        else:
            client = Client(api_server=API_SERVER_URLS[self.environment])
        client.create_project(self.project_name)
        return client

    def deploy_minimal_flow(
        self,
        flow_name: str = "ms2deepscore-minimal-flow",
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
        client = self._authenticate()

        mlflow_output_dir = MODEL_DIRECTORIES[self.environment]["mlflow"]

        flow_parameters = MinimalFlowParameters(
            model_uri=MODEL_DIRECTORIES[self.environment]["pre-trained-model"],
            input_dgw=self.input_dgw,
            overwrite=self.overwrite,
            environment=self.environment,
            spectrum_dgw=self.spectrum_dgw,
            overwrite_all=self.overwrite_all,
            redis_db=self.redis_db,
            n_chunks=self.n_chunks,
        )

        flow = build_minimal_flow(
            self.project_name,
            flow_name,
            self._make_flow_config(),
            flow_parameters,
            mlflow_output_dir=mlflow_output_dir,
            mlflow_server=self.mlflow_server,
            deploy_model=self.deploy_model,
        )
        flow_run_id = self._create_flow_run(client, flow)
        return flow_run_id

    def deploy_training_flow(
        self,
        flow_name: str = "ms2deepscore-training-flow",
    ):
        """
        Deploys a model training flow to Prefect cloud and optionally deploys it as a
        Seldon deployment.

        1) authenticate user credentials using the auth service endpoint
        2) process database and filesystem configs
        3) instantiate gateways and task parameters
        4) create prefect flow configuration
        5) builds the training flow graph
        6) pushes the flow to prefect cloud

        """

        client = self._authenticate()
        model_output_dir = MODEL_DIRECTORIES[self.environment]

        dataset_id = DATASET_IDS[self.environment][self.dataset_name].format(
            date=datetime.today()
        )

        flow_parameters = TrainingFlowParameters(
            input_dgw=self.input_dgw,
            environment=self.environment,
            ion_mode=self.ion_mode,
            spectrum_dgw=self.spectrum_dgw,
            dataset_id=dataset_id,
        )

        flow = build_training_flow(
            self.project_name,
            flow_name,
            self._make_flow_config(),
            flow_parameters,
            model_output_dir=model_output_dir,
            mlflow_server=self.mlflow_server,
            deploy_model=self.deploy_model,
        )

        flow_run_id = self._create_flow_run(client, flow)
        return flow_run_id

    def _create_flow_run(self, client, flow) -> str:
        flow_id = client.register(
            flow,
            project_name=self.project_name,
        )

        flow_run_id = client.create_flow_run(
            flow_id=flow_id,
            run_name=f"run {self.project_name}",
        )

        return flow_run_id

    def _make_flow_config(self):
        return make_flow_config(
            image=self.image,
            storage_type=PrefectStorageMethods.S3,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self.redis_db,
            environment=self.environment,
            schedule=self.schedule,
        )
