from datetime import datetime

from omigami.config import (
    MLFLOW_SERVER,
    DATASET_IDS,
    S3_BUCKETS,
)
from omigami.deployment import Deployer
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.ms2deepscore.config import MODEL_DIRECTORIES, PROJECT_NAME
from omigami.ms2deepscore.flows.pretrained_flow import (
    build_pretrained_flow,
    PretrainedFlowParameters,
)
from omigami.ms2deepscore.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
    ModelGeneralParameters,
)
from omigami.ms2deepscore.gateways.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)
from omigami.spectrum_cleaner import SpectrumCleaner


class MS2DeepScoreDeployer(Deployer):
    def __init__(
        self,
        mlflow_server: str = MLFLOW_SERVER,
        spectrum_ids_chunk_size: int = 1000,
        project_name: str = PROJECT_NAME,
        fingerprint_n_bits: int = 2048,
        scores_decimals: int = 5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._mlflow_server = mlflow_server
        self._spectrum_ids_chunk_size = spectrum_ids_chunk_size
        self._input_dgw = FSInputDataGateway()
        self._spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway()
        self._project_name = project_name
        self._fingerprint_n_bits = fingerprint_n_bits
        self._scores_decimals = scores_decimals

    def deploy_pretrained_flow(
        self,
        flow_name: str = "ms2deepscore-pretrained-flow",
    ):
        """
        Deploys a model pretrained flow to Prefect cloud and optionally deploys it as a
        Seldon deployment.

        1) authenticate user credentials using the auth service endpoint
        2) process database and filesystem configs
        3) instantiate gateways and task parameters
        4) create prefect flow configuration
        5) builds the pretrained flow graph
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
            overwrite_model=self._overwrite_model,
            environment=self._environment,
            spectrum_dgw=self._spectrum_dgw,
            overwrite_all_spectra=self._overwrite_all_spectra,
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
        output_dir = S3_BUCKETS[self._environment]

        dataset_id = DATASET_IDS[self._environment][self._dataset_name].format(
            date=datetime.today()
        )

        spectrum_cleaner = SpectrumCleaner()
        scores_output_path = MODEL_DIRECTORIES[self._environment]["scores"]

        flow_parameters = TrainingFlowParameters(
            input_dgw=self._input_dgw,
            environment=self._environment,
            ion_mode=self._ion_mode,
            spectrum_dgw=self._spectrum_dgw,
            output_dir=output_dir,
            dataset_id=dataset_id,
            chunk_size=self._spectrum_ids_chunk_size,
            overwrite_model=self._overwrite_model,
            overwrite_all_spectra=self._overwrite_all_spectra,
            source_uri=self._source_uri,
            spectrum_cleaner=spectrum_cleaner,
            scores_output_path=scores_output_path,
            fingerprint_n_bits=self._fingerprint_n_bits,
            scores_decimals=self._scores_decimals,
        )

        model_parameters = ModelGeneralParameters(
            model_output_dir=model_output_dir,
            mlflow_server=self._mlflow_server,
        )

        flow = build_training_flow(
            project_name=self._project_name,
            flow_name=flow_name,
            flow_config=self._make_flow_config(),
            flow_parameters=flow_parameters,
            model_parameters=model_parameters,
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
