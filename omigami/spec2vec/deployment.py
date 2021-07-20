from datetime import datetime

from omigami.config import (
    REDIS_DATABASES,
    S3_BUCKETS,
    DATASET_IDS,
)
from omigami.config import SOURCE_URI_PARTIAL_GNPS
from omigami.deployment import Deployer
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.input_data_gateway import FSInputDataGateway
from omigami.spec2vec.config import (
    MODEL_DIRECTORIES,
)
from omigami.spec2vec.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)
from omigami.spectrum_cleaner import SpectrumCleaner


class Spec2VecDeployer(Deployer):
    def __init__(
        self,
        iterations: int,
        window: int,
        intensity_weighting_power: float,
        allowed_missing_percentage: float,
        n_decimals: int = 2,
        chunk_size: int = int(1e8),
        source_uri: str = SOURCE_URI_PARTIAL_GNPS,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._iterations = iterations
        self._window = window
        self._intensity_weighting_power = intensity_weighting_power
        self._allowed_missing_percentage = allowed_missing_percentage
        self._n_decimals = n_decimals
        self._chunk_size = chunk_size
        self._source_uri = source_uri

    def deploy_training_flow(
        self,
        flow_name: str = "spec2vec-training-flow",
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
        client = self._authenticate()
        client.create_project(self._project_name)

        model_output_dir = MODEL_DIRECTORIES[self._environment]
        dataset_id = DATASET_IDS[self._environment][self._dataset_name].format(
            date=datetime.today()
        )
        redis_db = REDIS_DATABASES[self._environment][self._dataset_name]
        output_dir = S3_BUCKETS[self._environment]

        input_dgw = FSInputDataGateway()
        spectrum_dgw = Spec2VecRedisSpectrumDataGateway()
        spectrum_cleaner = SpectrumCleaner()

        flow_parameters = TrainingFlowParameters(
            input_dgw=input_dgw,
            spectrum_dgw=spectrum_dgw,
            spectrum_cleaner=spectrum_cleaner,
            source_uri=self._source_uri,
            output_dir=output_dir,
            dataset_id=dataset_id,
            chunk_size=self._chunk_size,
            ion_mode=self._ion_mode,
            n_decimals=self._n_decimals,
            overwrite_all_spectra=self._overwrite_all,
            iterations=self._iterations,
            window=self._window,
            overwrite_model=self._overwrite,
        )

        flow_config = make_flow_config(
            image=self._image,
            storage_type=PrefectStorageMethods.S3,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=redis_db,
            environment=self._environment,
            schedule=schedule,
        )

        flow = build_training_flow(
            self._project_name,
            flow_name,
            flow_config,
            flow_parameters,
            intensity_weighting_power=self._intensity_weighting_power,
            allowed_missing_percentage=self._allowed_missing_percentage,
            model_output_dir=model_output_dir,
            mlflow_server=self._mlflow_server,
            deploy_model=self._deploy_model,
        )

        training_flow_id = client.register(
            flow,
            project_name=self._project_name,
        )

        flow_run_id = client.create_flow_run(
            flow_id=training_flow_id,
            run_name=f"run {self._project_name}",
        )

        return flow_run_id
