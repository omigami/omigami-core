from datetime import datetime

from omigami.config import (
    MLFLOW_SERVER,
    DATASET_IDS,
)
from omigami.deployment import Deployer
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.config import MODEL_DIRECTORIES
from omigami.ms2deepscore.flows.pretrained_flow import (
    build_pretrained_flow,
    PretrainedFlowParameters,
)
from omigami.ms2deepscore.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


class MS2DeepScoreDeployer(Deployer):
    def __init__(
        self,
        mlflow_server: str = MLFLOW_SERVER,
        spectrum_ids_chunk_size: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mlflow_server = mlflow_server
        self._spectrum_ids_chunk_size = spectrum_ids_chunk_size
        self._input_dgw = FSInputDataGateway()
        self._spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()

    def deploy_pretrained_flow(
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
        client.create_project(self._project_name)

        if self._ion_mode != "positive":
            raise ValueError(
                f"Minimal flow can only be run with positive ion mode."
                f"{self._ion_mode} has been passed"
            )

        mlflow_output_dir = MODEL_DIRECTORIES[self._environment]["mlflow"]

        flow_parameters = PretrainedFlowParameters(
            model_uri=MODEL_DIRECTORIES[self._environment]["pre-trained-model"],
            input_dgw=self._input_dgw,
            overwrite=self._overwrite,
            environment=self._environment,
            spectrum_dgw=self._spectrum_dgw,
            overwrite_all=self._overwrite_all,
            redis_db=self._redis_db,
            spectrum_ids_chunk_size=self._spectrum_ids_chunk_size,
        )

        flow = build_pretrained_flow(
            self._project_name,
            flow_name,
            self._make_flow_config(),
            flow_parameters,
            mlflow_output_dir=mlflow_output_dir,
            mlflow_server=self._mlflow_server,
            deploy_model=self._deploy_model,
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
        client.create_project(self._project_name)

        model_output_dir = MODEL_DIRECTORIES[self._environment]

        dataset_id = DATASET_IDS[self._environment][self._dataset_name].format(
            date=datetime.today()
        )

        flow_parameters = TrainingFlowParameters(
            input_dgw=self._input_dgw,
            environment=self._environment,
            ion_mode=self._ion_mode,
            spectrum_dgw=self._spectrum_dgw,
            dataset_id=dataset_id,
        )

        flow = build_training_flow(
            self._project_name,
            flow_name,
            self._make_flow_config(),
            flow_parameters,
            model_output_dir=model_output_dir,
            mlflow_server=self._mlflow_server,
            deploy_model=self._deploy_model,
        )

        flow_run_id = self._create_flow_run(client, flow)
        return flow_run_id

    def _create_flow_run(self, client, flow) -> str:
        flow_id = client.register(
            flow,
            project_name=self._project_name,
        )

        flow_run_id = client.create_flow_run(
            flow_id=flow_id,
            run_name=f"run {self._project_name}",
        )

        return flow_run_id

    def _make_flow_config(self):
        return make_flow_config(
            image=self._image,
            storage_type=PrefectStorageMethods.S3,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_db,
            environment=self._environment,
            schedule=self._schedule,
        )
