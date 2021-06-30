from abc import ABC, abstractmethod
from typing import List

import mlflow
from mlflow.exceptions import MlflowException
from mlflow.pyfunc import PythonModel


class MLFlowModelRegister(ABC):
    def __init__(self, server_uri: str):
        mlflow.set_tracking_uri(server_uri)

    @staticmethod
    def _get_or_create_experiment_id(experiment_name: str, path: str = None) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name, artifact_location=path)

    @staticmethod
    def log_model(
        model: PythonModel,
        experiment_name: str,
        code_path: List[str],
        conda_env_path: str = None,
        output_path: str = None,
        **kwargs,
    ):
        try:
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name=experiment_name,
                conda_env=conda_env_path,
                code_path=code_path,
                **kwargs,
            )
        except MlflowException:
            mlflow.pyfunc.save_model(
                path=f"{output_path}/model",
                python_model=model,
                conda_env=conda_env_path,
                code_path=code_path,
                **kwargs,
            )

    @abstractmethod
    def register_model(
        self,
        model: PythonModel,
        experiment_name: str,
        output_path: str,
        conda_env_path: str = None,
    ):
        """
        Method to register to the MLFlow the model and the necessary metrics

        Parameters
        ----------
        model: PythonModel: Predictor that will hold the model and score the database
        experiment_name: MLFlow experiment name
        output_path: path to save the artifacts:
        conda_env_path: Conda environment requirements file
        """
        pass
