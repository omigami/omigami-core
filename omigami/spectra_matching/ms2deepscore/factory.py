from datetime import datetime
from typing import Dict, Optional

from drfs import DRPath
from prefect import Flow

from omigami.config import (
    REDIS_DATABASES,
    IonModes,
    DATASET_IDS,
    STORAGE_ROOT,
    CHUNK_SIZE,
    MLFLOW_DIRECTORY,
    MLFLOW_SERVER,
    GNPS_URIS,
)
from omigami.flow_config import make_flow_config, PrefectExecutorMethods
from omigami.spectra_matching.ms2deepscore.config import (
    DIRECTORIES,
    PROJECT_NAME,
    SPECTRUM_IDS_CHUNK_SIZE,
)
from omigami.spectra_matching.ms2deepscore.flows.deploy_model import (
    DeployModelFlowParameters,
    build_deploy_model_flow,
)
from omigami.spectra_matching.ms2deepscore.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.spectra_matching.ms2deepscore.storage.fs_data_gateway import (
    MS2DeepScoreFSDataGateway,
)
from omigami.spectra_matching.ms2deepscore.storage.redis_spectrum_gateway import (
    MS2DeepScoreRedisSpectrumDataGateway,
)


class MS2DeepScoreFlowFactory:
    def __init__(
        self,
        dataset_directory: str = None,
        mlflow_output_directory: str = None,
        directories: Dict[str, str] = None,
        model_registry_uri: str = None,
        storage_root: str = None,
    ):
        if storage_root is None:
            self._storage_root = STORAGE_ROOT
        else:
            self._storage_root = DRPath(storage_root)

        self._dataset_directory = (
            DRPath(dataset_directory)
            if dataset_directory is not None
            else self._storage_root / "datasets"
        )
        self._ms2deepscore_root = self._storage_root / "ms2deepscore"
        self._directories = directories or DIRECTORIES
        self._dataset_ids = DATASET_IDS
        self._model_registry_uri = model_registry_uri or MLFLOW_SERVER
        self._mlflow_output_directory = mlflow_output_directory or MLFLOW_DIRECTORY

    def build_training_flow(
        self,
        flow_name: str,
        dataset_id: str,
        fingerprint_n_bits: int,
        scores_decimals: int,
        spectrum_binner_n_bins: int,
        image: Optional[str] = None,
        schedule: int = None,
        ion_mode: IonModes = "positive",
        project_name: str = PROJECT_NAME,
        spectrum_ids_chunk_size: int = SPECTRUM_IDS_CHUNK_SIZE,
        train_ratio: float = 0.9,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        epochs: int = 50,
        chunk_size: int = CHUNK_SIZE,
    ) -> Flow:
        """Creates all configuration/gateways objects used by the training flow, and builds
        the training flow with them.

        Parameters
        ----------
        For information on parameters please check omigami/ms2deepscore/cli.py

        Returns
        -------
        Flow:
            A prefect training flow with the given parameters

        """
        flow_config = make_flow_config(
            image=image,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=REDIS_DATABASES[dataset_id],
            schedule=schedule,
            storage_root=self._storage_root,
        )

        fs_dgw = MS2DeepScoreFSDataGateway()

        spectrum_binner_output_path = self._ms2deepscore_root / self._directories[
            "spectrum_binner"
        ].format(dataset_id=dataset_id)
        binned_spectra_path = self._ms2deepscore_root / self._directories[
            "binned_spectra"
        ].format(dataset_id=dataset_id)
        model_output_path = str(self._ms2deepscore_root / self._directories["model"])
        scores_output_path = self._ms2deepscore_root / self._directories["scores"]

        source_uri = GNPS_URIS[dataset_id]
        dataset_id = self._dataset_ids[dataset_id].format(date=datetime.today())
        flow_parameters = TrainingFlowParameters(
            fs_dgw=fs_dgw,
            source_uri=source_uri,
            dataset_directory=self._dataset_directory / dataset_id,
            chunk_size=chunk_size,
            ion_mode=ion_mode,
            scores_output_path=str(scores_output_path),
            fingerprint_n_bits=fingerprint_n_bits,
            scores_decimals=scores_decimals,
            spectrum_binner_output_path=spectrum_binner_output_path,
            binned_spectra_output_path=binned_spectra_path,
            spectrum_binner_n_bins=spectrum_binner_n_bins,
            model_output_path=model_output_path,
            project_name=project_name,
            model_registry_uri=self._model_registry_uri,
            mlflow_output_directory=str(self._mlflow_output_directory),
            epochs=epochs,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            spectrum_ids_chunk_size=spectrum_ids_chunk_size,
            schedule_task_days=schedule,
        )

        ms2deepscore_flow = build_training_flow(
            flow_name=flow_name,
            flow_config=flow_config,
            flow_parameters=flow_parameters,
        )

        return ms2deepscore_flow

    def build_model_deployment_flow(
        self,
        flow_name: str,
        image: str,
        dataset_id: str,
        ion_mode: IonModes = "positive",
        project_name: str = PROJECT_NAME,
    ) -> Flow:
        """Creates all configuration/gateways objects used by the model deployment flow,
        and builds the training flow with them.

        Parameters
        ----------
        For information on parameters please check omigami/ms2deepscore/cli.py

        Returns
        -------
        Flow:
            A prefect model deployment flow with the given parameters

        """
        flow_config = make_flow_config(
            image=image,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=REDIS_DATABASES[dataset_id],
            storage_root=self._storage_root,
        )

        spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway(project=project_name)
        fs_dgw = MS2DeepScoreFSDataGateway()

        dataset_id = self._dataset_ids[dataset_id].format(date=datetime.today())
        dataset_directory = self._dataset_directory / dataset_id
        flow_parameters = DeployModelFlowParameters(
            spectrum_dgw=spectrum_dgw,
            fs_dgw=fs_dgw,
            ion_mode=ion_mode,
            redis_db=REDIS_DATABASES[dataset_id],
            model_registry_uri=self._model_registry_uri,
            dataset_directory=dataset_directory,
        )

        deploy_model_flow = build_deploy_model_flow(
            flow_name,
            flow_config,
            flow_parameters,
        )

        return deploy_model_flow
