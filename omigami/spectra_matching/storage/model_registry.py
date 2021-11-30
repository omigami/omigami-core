from abc import ABC, abstractmethod
from typing import Any, Optional, Dict

import mlflow

from omigami.config import CODE_PATH, MLFLOW_SERVER
from omigami.spectra_matching.predictor import Predictor


class ModelRegistryDataGateway(ABC):
    @abstractmethod
    def register_model(
        self,
        model: Predictor,
        experiment_name: str,
        run_name: str,
        output_path: str = None,
        conda_env_path: str = None,
        model_name: Optional[str] = None,
        params: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        artifacts: Dict[str, Any] = None,
    ):
        """TODO"""
        pass


class MLFlowDataGateway(ModelRegistryDataGateway):
    def __init__(self, tracking_uri: str = None):
        mlflow.set_tracking_uri(tracking_uri or MLFLOW_SERVER)

    def register_model(
        self,
        model: Predictor,
        experiment_name: str,
        run_name: str,
        output_path: str = None,
        conda_env_path: str = None,
        model_name: Optional[str] = None,
        params: Dict[str, Any] = None,
        metrics: Dict[str, float] = None,
        artifacts: Dict[str, Any] = None,
    ) -> str:
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment_id, nested=True
        ) as run:
            if params is not None:
                mlflow.log_params(params)
            if metrics is not None:
                _ = [mlflow.log_metric(k, v) for k, v in metrics.items()]

            run_id = run.info.run_id
            model.set_run_id(run_id)
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name=model_name,
                code_path=[CODE_PATH],
                conda_env=conda_env_path,
                artifacts=artifacts,
            )

            return run_id

    @staticmethod
    def _get_or_create_experiment_id(experiment_name: str, path: str = None) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name, artifact_location=path)
