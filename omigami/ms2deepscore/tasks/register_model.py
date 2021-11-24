from dataclasses import dataclass
from typing import Dict

import mlflow
from pandas import Timestamp
from prefect import Task

from omigami.config import IonModes
from omigami.model_register import MLFlowModelRegister
from omigami.ms2deepscore.helper_classes.siamese_model_trainer import (
    SIAMESE_MODEL_PARAMS,
)
from omigami.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.ms2deepscore.tasks.train_model import TrainModelParameters
from omigami.utils import merge_prefect_task_configs

CONDA_ENV_PATH = "./requirements/environment.frozen.yaml"


@dataclass
class RegisterModelParameters:
    experiment_name: str
    mlflow_output_path: str
    server_uri: str
    ion_mode: IonModes


class RegisterModel(Task):
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

    def run(
        self,
        train_model_output: dict = None,
    ) -> Dict[str, str]:
        """
        Prefect task to register an ms2deepscore model to MLflow Model Registry. Model
        parameters saved during registration:
            - `epochs`
            - `learning_rate`
            - `layer_base_dims`
            - `embedding_dim`
            - `dropout_rate`
            - `split_ratio`
            - `validation_loss`

        Parameters
        ----------
        train_model_output: Dict[str, str]
            Dictionary containing `ms2deepscore_model_path` and `validation_loss`

        Returns
        -------
        Dictionary containing registered `model_uri` and `run_id`

        """
        model_path = train_model_output["ms2deepscore_model_path"]
        validation_loss = train_model_output["validation_loss"]

        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._mlflow_output_path}."
        )
        run_name = f"ms2deepscore-{self._ion_mode}-{Timestamp.now():%Y%m%dT%H%M}"

        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            model=MS2DeepScorePredictor(self._ion_mode),
            experiment_name=self._experiment_name,
            output_path=self._mlflow_output_path,
            train_parameters=self.training_parameters,
            validation_loss=validation_loss,
            conda_env_path=CONDA_ENV_PATH,
            artifacts={"ms2deepscore_model_path": model_path},
            run_name=run_name,
        )
        run = mlflow.get_run(run_id)
        self.logger.info(f"{run.info}")

        model_uri = f"{run.info.artifact_uri}/model/"
        return {"model_uri": model_uri, "run_id": run_id}


class ModelRegister(MLFlowModelRegister):
    """
    Class that implements MLFLowModelRegister to register ms2deepscore model to MLFlow
    """

    # TODO: should be refactored together with spec2vec's implementation of this code
    def register_model(
        self,
        model: MS2DeepScorePredictor,
        experiment_name: str,
        output_path: str,
        train_parameters: TrainModelParameters = None,
        validation_loss: float = None,
        conda_env_path: str = None,
        artifacts: Dict = None,
        run_name: str = "ms2deepscore",
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
        train_parameters: TrainModelParameters = None
            Training Parameters used to train the model
        validation_loss: float = None
            Last validation loss of the training
        conda_env_path: str = None
            Conda environment requirements file
        artifacts: Dict = None
            Dictionary of artifacts to be stored along with the model
        run_name: str = "ms2deepscore"
            Name of the run to identify in the MLFlow
        Returns
        -------
            Return the MLFLow run ID
        """
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment_id, nested=True
        ) as run:
            run_id = run.info.run_id
            model.set_run_id(run_id)

            if train_parameters:
                mlflow.log_params(
                    self._convert_train_parameters(train_parameters, validation_loss)
                )

            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name="ms2deepscore",
                code_path=["omigami"],
                conda_env=conda_env_path,
                artifacts=artifacts,
            )

            return run_id

    @staticmethod
    def _convert_train_parameters(
        train_parameters: TrainModelParameters, validation_loss: float
    ) -> Dict:
        """Converts training parameters and the models metrics into a dict"""

        train_params = {
            "epochs": train_parameters.epochs,
            "learning_rate": SIAMESE_MODEL_PARAMS["learning_rate"],
            "layer_base_dims": SIAMESE_MODEL_PARAMS["layer_base_dims"],
            "embedding_dim": SIAMESE_MODEL_PARAMS["embedding_dim"],
            "dropout_rate": SIAMESE_MODEL_PARAMS["dropout_rate"],
            "split_ratio": train_parameters.split_ratio,
            "validation_loss": validation_loss,
        }

        return train_params
