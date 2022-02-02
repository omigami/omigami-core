from dataclasses import dataclass
from typing import Dict

from pandas import Timestamp
from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.ms2deepscore.config import PREDICTOR_ENV_PATH
from omigami.spectra_matching.ms2deepscore.helper_classes.siamese_model_trainer import (
    SIAMESE_MODEL_PARAMS,
)
from omigami.spectra_matching.ms2deepscore.predictor import MS2DeepScorePredictor
from omigami.spectra_matching.ms2deepscore.tasks.train_model import TrainModelParameters
from omigami.spectra_matching.storage.model_registry import (
    MLFlowDataGateway,
)
from omigami.utils import merge_prefect_task_configs


@dataclass
class RegisterModelParameters:
    experiment_name: str
    model_registry_uri: str
    mlflow_output_path: str
    ion_mode: IonModes


class RegisterModel(Task):
    def __init__(
        self,
        parameters: RegisterModelParameters,
        training_parameters: TrainModelParameters = None,
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._model_registry_uri = parameters.model_registry_uri
        self._mlflow_output_path = parameters.mlflow_output_path
        self._ion_mode = parameters.ion_mode
        self.training_parameters = training_parameters
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, train_model_output: dict = None) -> str:
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
        Model `run_id`

        """
        self.logger.info(
            f"Registering model on URI {self._model_registry_uri} on path: "
            f"{self._mlflow_output_path}."
        )
        params = self._convert_train_parameters(
            train_model_output.pop("validation_loss")
        )

        run_name = f"ms2deepscore-{self._ion_mode}-{Timestamp.now():%Y%m%dT%H%M}"

        mlflow_dgw = MLFlowDataGateway(self._model_registry_uri)

        run_id = mlflow_dgw.register_model(
            model=MS2DeepScorePredictor(self._ion_mode),
            conda_env_path=PREDICTOR_ENV_PATH,
            experiment_name=self._experiment_name,
            experiment_path=self._mlflow_output_path,
            params=params,
            artifacts=train_model_output,
            run_name=run_name,
            model_name=f"MS2DeepScore-{self._ion_mode}",
        )

        self.logger.info(f"Created model run_id: {run_id}.")

        return run_id

    def _convert_train_parameters(self, validation_loss: float) -> Dict[str, float]:
        """Converts training parameters and the models metrics into a dict"""

        train_params = {
            "epochs": self.training_parameters.epochs,
            "learning_rate": SIAMESE_MODEL_PARAMS["learning_rate"],
            "layer_base_dims": SIAMESE_MODEL_PARAMS["layer_base_dims"],
            "embedding_dim": SIAMESE_MODEL_PARAMS["embedding_dim"],
            "dropout_rate": SIAMESE_MODEL_PARAMS["dropout_rate"],
            "split_ratio": self.training_parameters.split_ratio,
            "validation_loss": validation_loss,
        }

        return train_params
