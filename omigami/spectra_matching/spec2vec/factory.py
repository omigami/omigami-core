from datetime import datetime
from typing import Dict

import pandas as pd
from drfs import DRPath
from prefect import Flow

from omigami.config import (
    REDIS_DATABASES,
    IonModes,
    DATASET_IDS,
    MLFLOW_DIRECTORY,
    STORAGE_ROOT,
    CHUNK_SIZE,
    MLFLOW_SERVER,
    GNPS_URIS,
)
from omigami.flow_config import (
    make_flow_config,
    PrefectExecutorMethods,
)
from omigami.spectra_matching.spec2vec.config import (
    DOCUMENT_DIRECTORIES,
    SPEC2VEC_ROOT,
)
from omigami.spectra_matching.spec2vec.flows.deploy_model import (
    DeployModelFlowParameters,
    build_deploy_model_flow,
)
from omigami.spectra_matching.spec2vec.flows.training_flow import (
    TrainingFlowParameters,
    build_training_flow,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway


class Spec2VecFlowFactory:
    def __init__(
        self,
        dataset_directory: str = None,
        documents_dir: Dict[str, str] = None,
        model_registry_uri: str = None,
        mlflow_output_directory: str = None,
    ):
        self._redis_dbs = REDIS_DATABASES
        self._dataset_directory = (
            DRPath(dataset_directory)
            if dataset_directory is not None
            else STORAGE_ROOT / "datasets"
        )
        self._spec2vec_root = SPEC2VEC_ROOT
        self._model_registry_uri = model_registry_uri or MLFLOW_SERVER
        self._mlflow_output_directory = mlflow_output_directory or str(MLFLOW_DIRECTORY)
        self._document_dirs = documents_dir or DOCUMENT_DIRECTORIES
        self._dataset_ids = DATASET_IDS

    def build_training_flow(
        self,
        project_name: str,
        flow_name: str,
        image: str,
        iterations: int,
        window: int,
        intensity_weighting_power: float,
        allowed_missing_percentage: float,
        dataset_id: str,
        n_decimals: int = 2,
        schedule: pd.Timedelta = None,
        ion_mode: IonModes = "positive",
        overwrite_model: bool = False,
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
            A prefect training flow built with the given parameters

        """
        flow_config = make_flow_config(
            image=image,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_dbs[dataset_id],
            schedule=schedule,
            storage_root=STORAGE_ROOT,
        )

        spectrum_dgw = RedisSpectrumDataGateway(project_name)
        fs_dgw = FSDataGateway()

        source_uri = GNPS_URIS[dataset_id]
        dataset_id = self._dataset_ids[dataset_id].format(date=datetime.today())
        flow_parameters = TrainingFlowParameters(
            fs_dgw=fs_dgw,
            spectrum_dgw=spectrum_dgw,
            dataset_directory=f"{self._dataset_directory}/{dataset_id}",
            ion_mode=ion_mode,
            n_decimals=n_decimals,
            iterations=iterations,
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
            window=window,
            chunk_size=chunk_size,
            source_uri=source_uri,
            overwrite_model=overwrite_model,
            documents_save_directory=str(
                self._spec2vec_root
                / self._document_dirs[ion_mode]
                / dataset_id
                / f"{n_decimals}_decimals"
            ),
            model_registry_uri=self._model_registry_uri,
            mlflow_output_directory=self._mlflow_output_directory,
            experiment_name=project_name,
            redis_db=self._redis_dbs[dataset_id],
        )

        training_flow = build_training_flow(
            flow_name,
            flow_config,
            flow_parameters,
            deploy_model=deploy_model,
        )

        return training_flow

    def build_model_deployment_flow(
        self,
        project_name: str,
        flow_name: str,
        image: str,
        intensity_weighting_power: float,
        allowed_missing_percentage: float,
        dataset_id: str,
        n_decimals: int = 2,
        ion_mode: IonModes = "positive",
    ) -> Flow:
        """Creates all configuration/gateways objects used by the model deployment flow,
        and builds the training flow with them.

        Parameters
        ----------
        For information on parameters please check omigami/spec2vec/cli.py

        Returns
        -------
        Flow:
            A prefect model deployment flow with the given parameters

        """
        flow_config = make_flow_config(
            image=image,
            executor_type=PrefectExecutorMethods.LOCAL_DASK,
            redis_db=self._redis_dbs[dataset_id],
            storage_root=STORAGE_ROOT,
        )

        spectrum_dgw = RedisSpectrumDataGateway(project_name)
        fs_dgw = FSDataGateway()

        dataset_id = self._dataset_ids[dataset_id].format(date=datetime.today())
        flow_parameters = DeployModelFlowParameters(
            spectrum_dgw=spectrum_dgw,
            fs_dgw=fs_dgw,
            ion_mode=ion_mode,
            n_decimals=n_decimals,
            documents_directory=str(
                self._spec2vec_root / dataset_id / self._document_dirs[ion_mode]
            ),
            intensity_weighting_power=intensity_weighting_power,
            allowed_missing_percentage=allowed_missing_percentage,
            redis_db=self._redis_dbs[dataset_id],
            model_registry_uri=self._model_registry_uri,
        )

        deploy_model_flow = build_deploy_model_flow(
            flow_name,
            flow_config,
            flow_parameters,
        )

        return deploy_model_flow
