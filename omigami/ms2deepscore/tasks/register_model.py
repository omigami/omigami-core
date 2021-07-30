from dataclasses import dataclass
from typing import Dict

import mlflow
from mlflow.pyfunc import PythonModel
from omigami.config import IonModes
from omigami.model_register import MLFlowModelRegister
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.utils import merge_prefect_task_configs
from prefect import Task

CONDA_ENV_PATH = "./requirements/environment.frozen.yaml"


@dataclass
class RegisterModelParameters:
    experiment_name: str
    mlflow_output_path: str
    server_uri: str
    ion_mode: IonModes


class RegisterModel(Task):
    """
    Prefect task to register a model to MLFlow
    """

    def __init__(
        self,
        parameters: RegisterModelParameters,
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._mlflow_output_path = parameters.mlflow_output_path
        self._server_uri = parameters.server_uri
        self._ion_mode = parameters.ion_mode

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_path: str = None) -> Dict[str, str]:
        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._mlflow_output_path}."
        )

        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            MS2DeepScorePredictor(self._ion_mode),
            self._experiment_name,
            self._mlflow_output_path,
            CONDA_ENV_PATH,
            artifacts={"ms2deepscore_model_path": model_path},
        )
        run = mlflow.get_run(run_id)
        self.logger.info(f"{run.info}")

        model_uri = f"{run.info.artifact_uri}/model/"
        return {"model_uri": model_uri, "run_id": run_id}


class ModelRegister(MLFlowModelRegister):
    """
    Class that implements MLFLowModelRegister to register ms2deepscore model to MLFlow
    """

    def register_model(
        self,
        model: PythonModel,
        experiment_name: str,
        output_path: str,
        conda_env_path: str = None,
        artifacts: Dict = None,
        **kwargs,
    ):
        """
        Method to register the MS2DeepScore to MLFlow.

        Parameters
        ----------
        model: PythonModel class to execute the predictions
        experiment_name: MLFlow Experiment name
        output_path: path to save the artifacts
        conda_env_path: Conda environment requirements file
        artifacts: Dictionary of artifacts to be stored along with the model

        Returns
        -------
            Return the MLFLow run ID
        """
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
            run_id = run.info.run_id

            self.log_model(
                model,
                experiment_name,
                output_path=output_path,
                code_path=[
                    "omigami",
                ],
                conda_env_path=conda_env_path,
                artifacts=artifacts,
                **kwargs,
            )

            return run_id
