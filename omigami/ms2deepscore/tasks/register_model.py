from dataclasses import dataclass
from typing import Dict, Any

import mlflow
from mlflow.pyfunc import PythonModel

from omigami.config import IonModes
from omigami.model_register import MLFlowModelRegister
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.ms2deepscore.tasks.train_model import TrainModelParameters
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
        training_parameters: TrainModelParameters = None,
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._mlflow_output_path = parameters.mlflow_output_path
        self._server_uri = parameters.server_uri
        self._ion_mode = parameters.ion_mode
        self.training_parameters = training_parameters
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_path: str = None) -> Dict[str, str]:
        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._mlflow_output_path}."
        )

        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            model=MS2DeepScorePredictor(self._ion_mode),
            experiment_name=self._experiment_name,
            output_path=self._mlflow_output_path,
            train_parameters=self.training_parameters,
            conda_env_path=CONDA_ENV_PATH,
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
        model: MS2DeepScorePredictor,
        experiment_name: str,
        output_path: str,
        train_parameters: TrainModelParameters = None,
        conda_env_path: str = None,
        artifacts: Dict = None,
        **kwargs,
    ):
        """
        Method to register the MS2DeepScore to MLFlow.

        Parameters
        ----------
        model: MS2DeepScorePredictor
            PythonModel class to execute the predictions
        experiment_name: str
            MLFlow Experiment name
        output_path: str
            Path to save the artifacts
        train_parameters: TrainModelParameters
            Training Parameters used to train the model
        conda_env_path: str = None
            Conda environment requirements file
        artifacts: Dict = None
            Dictionary of artifacts to be stored along with the model

        Returns
        -------
            Return the MLFLow run ID
        """
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
            run_id = run.info.run_id

            if train_parameters:
                mlflow.log_params(
                    self._convert_train_parameters(model, train_parameters)
                )

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

    @staticmethod
    def _convert_train_parameters(
        model: MS2DeepScorePredictor, train_parameters: TrainModelParameters
    ) -> Dict:
        """Converts training parameters and the models metrics into a dict"""

        train_params = {
            "spectrum_binner_output_path": train_parameters.spectrum_binner_output_path,
            "epochs": train_parameters.epochs,
            "learning_rate": train_parameters.learning_rate,
            "layer_base_dims": train_parameters.layer_base_dims,
            "embedding_dim": train_parameters.embedding_dim,
            "dropout_rate": train_parameters.dropout_rate,
            "split_ratio": train_parameters.split_ratio,
        }

        # TODO: At this point the model is always None. Becuase it is never assigned.
        if model.model:
            model_metrics = model.model.model.history.history
            del model_metrics["loss"]
            train_params.update(model_metrics)

        return train_params
