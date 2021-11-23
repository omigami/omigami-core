from datetime import datetime
from typing import Dict

import pandas as pd
from prefect import Flow

from omigami.config import (
    REDIS_DATABASES,
    IonModes,
    DATASET_IDS,
    MLFLOW_SERVER,
    STORAGE_ROOT,
    CHUNK_SIZE,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.config import (
    DIRECTORIES,
    PROJECT_NAME,
    SPECTRUM_IDS_CHUNK_SIZE,
    MS2DEEPSCORE_ROOT,
)
from omigami.ms2deepscore.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.ms2deepscore.gateways import MS2DeepScoreRedisSpectrumDataGateway
from omigami.spectrum_cleaner import SpectrumCleaner


class MS2DeepScoreFlowFactory:
    def __init__(
        self,
        dataset_directory: str = None,
        directories: Dict[str, str] = None,
    ):
        self._redis_dbs = REDIS_DATABASES
        self._dataset_directory = dataset_directory or STORAGE_ROOT / "datasets"
        self._ms2deepscore_root = MS2DEEPSCORE_ROOT
        self._directories = directories or DIRECTORIES
        self._dataset_ids = DATASET_IDS
        self._mlflow_server = MLFLOW_SERVER
        self._storage_type = (
            PrefectStorageMethods.S3
            if "s3" in str(MS2DEEPSCORE_ROOT)
            else PrefectStorageMethods.Local
        )

    def build_training_flow(
        self,
        flow_name: str,
        image: str,
        dataset_name: str,
        fingerprint_n_bits: int,
        scores_decimals: int,
        source_uri: str,
        spectrum_binner_n_bins: int,
        schedule: pd.Timedelta = None,
        ion_mode: IonModes = "positive",
        deploy_model: bool = False,
        overwrite_all_spectra: bool = False,
        overwrite_model: bool = False,
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
            storage_type=self._storage_type,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_dbs[dataset_name],
            schedule=schedule,
            storage_root=STORAGE_ROOT,
        )

        spectrum_dgw = MS2DeepScoreRedisSpectrumDataGateway(project=project_name)
        data_gtw = FSDataGateway()
        spectrum_cleaner = SpectrumCleaner()

        spectrum_binner_output_path = (
            self._ms2deepscore_root / self._directories["spectrum_binner"]
        )
        model_output_path = self._ms2deepscore_root / self._directories["model"]
        mlflow_dir = self._ms2deepscore_root / self._directories["mlflow"]
        scores_output_path = self._ms2deepscore_root / self._directories["scores"]

        flow_parameters = TrainingFlowParameters(
            data_gtw=data_gtw,
            ion_mode=ion_mode,
            spectrum_dgw=spectrum_dgw,
            dataset_directory=self._dataset_directory,
            chunk_size=chunk_size,
            dataset_id=self._dataset_ids[dataset_name].format(date=datetime.today()),
            overwrite_model=overwrite_model,
            overwrite_all_spectra=overwrite_all_spectra,
            source_uri=source_uri,
            spectrum_cleaner=spectrum_cleaner,
            scores_output_path=scores_output_path,
            fingerprint_n_bits=fingerprint_n_bits,
            scores_decimals=scores_decimals,
            spectrum_binner_output_path=spectrum_binner_output_path,
            spectrum_binner_n_bins=spectrum_binner_n_bins,
            model_output_path=model_output_path,
            epochs=epochs,
            train_ratio=train_ratio,
            validation_ratio=validation_ratio,
            test_ratio=test_ratio,
            project_name=project_name,
            mlflow_output_dir=mlflow_dir,
            mlflow_server=self._mlflow_server,
            redis_db=self._redis_dbs[dataset_name],
            spectrum_ids_chunk_size=spectrum_ids_chunk_size,
        )

        ms2deepscore_flow = build_training_flow(
            flow_name=flow_name,
            flow_config=flow_config,
            flow_parameters=flow_parameters,
            deploy_model=deploy_model,
        )

        return ms2deepscore_flow

    def build_model_deployment_flow(self) -> Flow:
        """TODO"""
        raise NotImplemented
