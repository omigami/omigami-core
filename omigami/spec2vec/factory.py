from datetime import datetime
from typing import Dict

import pandas as pd
from prefect import Flow

from omigami.config import (
    REDIS_DATABASES,
    IonModes,
    DATASET_IDS,
    MLFLOW_SERVER,
    MLFLOW_DIRECTORY,
    STORAGE_ROOT,
    CHUNK_SIZE,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.spec2vec.config import (
    PROJECT_NAME,
    DOCUMENT_DIRECTORIES,
    SPEC2VEC_ROOT,
)
from omigami.spec2vec.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.spec2vec.gateways.redis_spectrum_document import (
    RedisSpectrumDocumentDataGateway,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway
from omigami.spectra_matching.spectrum_cleaner import SpectrumCleaner


class Spec2VecFlowFactory:
    def __init__(
        self,
        dataset_directory: str = None,
        documents_dir: Dict[str, str] = None,
        mlflow_output_directory: str = None,
    ):
        self._redis_dbs = REDIS_DATABASES
        self._dataset_directory = dataset_directory or str(STORAGE_ROOT / "datasets")
        self._spec2vec_root = SPEC2VEC_ROOT
        self._mlflow_output_directory = mlflow_output_directory or str(MLFLOW_DIRECTORY)
        self._document_dirs = documents_dir or DOCUMENT_DIRECTORIES
        self._dataset_ids = DATASET_IDS
        self._mlflow_server = MLFLOW_SERVER
        self._storage_type = (
            PrefectStorageMethods.S3
            if "s3" in str(SPEC2VEC_ROOT)
            else PrefectStorageMethods.Local
        )

    def build_training_flow(
        self,
        flow_name: str,
        image: str,
        iterations: int,
        window: int,
        intensity_weighting_power: float,
        allowed_missing_percentage: float,
        dataset_id: str,
        source_uri: str,
        n_decimals: int = 2,
        schedule: pd.Timedelta = None,
        ion_mode: IonModes = "positive",
        overwrite_all_spectra: bool = False,
        overwrite_model: bool = False,
        project_name: str = PROJECT_NAME,
        deploy_model: bool = False,
        chunk_size: int = CHUNK_SIZE,
    ) -> Flow:
        """Creates all configuration/gateways objects used by the training flow, and builds
        the training flow with them.

        Parameters
        ----------
        For information on parameters please check omigami/spec2vec/cli.py

        Returns
        -------
        Flow:
            A prefect training flow with the given parameters

        """
        flow_config = make_flow_config(
            image=image,
            storage_type=self._storage_type,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_dbs[dataset_id],
            schedule=schedule,
            storage_root=STORAGE_ROOT,
        )

        spectrum_dgw = RedisSpectrumDataGateway(project=project_name)
        data_gtw = FSDataGateway()
        spectrum_cleaner = SpectrumCleaner()
        document_dgw = RedisSpectrumDocumentDataGateway()

        flow_parameters = TrainingFlowParameters(
            data_gtw=data_gtw,
            spectrum_dgw=spectrum_dgw,
            document_dgw=document_dgw,
            spectrum_cleaner=spectrum_cleaner,
            dataset_id=self._dataset_ids[dataset_id].format(date=datetime.today()),
            ion_mode=ion_mode,
            n_decimals=n_decimals,
            iterations=iterations,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
            window=window,
            chunk_size=chunk_size,
            source_uri=source_uri,
            overwrite_model=overwrite_model,
            overwrite_all_spectra=overwrite_all_spectra,
            documents_save_directory=str(
                self._spec2vec_root / self._document_dirs[ion_mode]
            ),
            dataset_directory=self._dataset_directory,
            mlflow_output_directory=self._mlflow_output_directory,
            mlflow_server=self._mlflow_server,
            experiment_name=project_name,
            redis_db=self._redis_dbs[dataset_id],
        )

        spec2vec_flow = build_training_flow(
            flow_name,
            flow_config,
            flow_parameters,
            deploy_model=deploy_model,
        )

        return spec2vec_flow

    def build_model_deployment_flow(self) -> Flow:
        """TODO"""
        raise NotImplemented
