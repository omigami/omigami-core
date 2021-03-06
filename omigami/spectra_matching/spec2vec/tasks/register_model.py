from dataclasses import dataclass
from typing import Union, Optional

from pandas import Timestamp
from prefect import Task

from omigami.config import IonModes
from omigami.spectra_matching.spec2vec.config import PREDICTOR_ENV_PATH
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.spectra_matching.spec2vec.tasks.train_model import TrainModelParameters
from omigami.spectra_matching.storage.model_registry import MLFlowDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class RegisterModelParameters:
    experiment_name: str
    model_registry_uri: str
    mlflow_output_directory: str
    n_decimals: int
    ion_mode: IonModes
    intensity_weighting_power: Union[float, int]
    allowed_missing_percentage: Union[float, int]
    model_name: Optional[str]


class RegisterModel(Task):
    def __init__(
        self,
        parameters: RegisterModelParameters,
        training_params: TrainModelParameters,
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._model_registry_uri = parameters.model_registry_uri
        self._mlflow_output_directory = parameters.mlflow_output_directory
        self._n_decimals = parameters.n_decimals
        self._ion_mode = parameters.ion_mode
        self._intensity_weighting_power = parameters.intensity_weighting_power
        self._allowed_missing_percentage = parameters.allowed_missing_percentage
        self._model_name = parameters.model_name
        self._training_parameters = training_params
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model_path: str = None) -> str:
        """
        Prefect task to register the model to MLflow Model Registry. `alpha` is saved as
        model metric. Following are saved as model parameters:
            - `n_decimals_for_documents`
            - `intensity_weighting_power`
            - `allowed_missing_percentage`
            - `iter`
            - `window`

        Parameters
        ----------
        model_path: str
            Model trained on spectrum documents

        Returns
        -------
        Dictionary containing registered model's `model_uri` and `run_id`

        """
        self.logger.info(
            f"Registering model to {self._model_registry_uri} on URI: "
            f"{self._mlflow_output_directory}."
        )
        run_name = f"spec2vec-{self._ion_mode}-{Timestamp.now():%Y%m%dT%H%M}"
        model_register = MLFlowDataGateway(self._model_registry_uri)
        artifacts = {"word2vec_model": model_path}
        spec2vec_model = Spec2VecPredictor(
            self._ion_mode,
            self._n_decimals,
            self._intensity_weighting_power,
            self._allowed_missing_percentage,
        )

        params = {
            "n_decimals_for_documents": self._n_decimals,
            "intensity_weighting_power": self._intensity_weighting_power,
            "allowed_missing_percentage": self._allowed_missing_percentage,
            "iter": self._training_parameters.epochs,
            "window": self._training_parameters.window,
            "ion_mode": self._ion_mode,
        }

        run_id = model_register.register_model(
            spec2vec_model,
            experiment_name=self._experiment_name,
            conda_env_path=PREDICTOR_ENV_PATH,
            experiment_path=self._mlflow_output_directory,
            run_name=run_name,
            model_name=self._model_name,
            params=params,
            artifacts=artifacts,
        )

        self.logger.info(f"Created model run_id: {run_id}.")

        return run_id
