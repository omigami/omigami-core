from dataclasses import dataclass
from typing import Union, Dict, Optional

import mlflow
from gensim.models import Word2Vec
from pandas import Timestamp
from prefect import Task

from omigami.config import IonModes, CONDA_ENV_PATH, CODE_PATH, MLFLOW_SERVER
from omigami.spectra_matching.model_register import MLFlowModelRegister
from omigami.spectra_matching.spec2vec.predictor import Spec2VecPredictor
from omigami.utils import merge_prefect_task_configs


@dataclass
class RegisterModelParameters:
    experiment_name: str
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
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._path = parameters.mlflow_output_directory
        self._n_decimals = parameters.n_decimals
        self._ion_mode = parameters.ion_mode
        self._intensity_weighting_power = parameters.intensity_weighting_power
        self._allowed_missing_percentage = parameters.allowed_missing_percentage
        self._model_name = parameters.model_name
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model: Word2Vec = None) -> Dict[str, str]:
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
        model: Word2Vec
            Model trained on spectrum documents

        Returns
        -------
        Dictionary containing registered model's `model_uri` and `run_id`

        """
        self.logger.info(f"Registering model to {MLFLOW_SERVER} on URI: {self._path}.")
        run_name = f"spec2vec-{Timestamp.now():%Y%m%dT%H%M}"
        model_register = ModelRegister(MLFLOW_SERVER)
        run_id = model_register.register_model(
            Spec2VecPredictor(
                model,
                self._ion_mode,
                self._n_decimals,
                self._intensity_weighting_power,
                self._allowed_missing_percentage,
            ),
            experiment_name=self._experiment_name,
            output_path=self._path,
            conda_env_path=CONDA_ENV_PATH,
            run_name=run_name,
            model_name=self._model_name,
        )
        run = mlflow.get_run(run_id)
        self.logger.info(f"{run.info}")

        model_uri = f"{run.info.artifact_uri}/model/"
        return {"model_uri": model_uri, "run_id": run_id}


class ModelRegister(MLFlowModelRegister):
    # TODO: refactor this class. The concrete implementation should inherit the
    # TODO: abstraction, and here it is the other way around???
    def register_model(
        self,
        model: Spec2VecPredictor,
        experiment_name: str,
        output_path: str = None,
        conda_env_path: str = None,
        run_name: str = "spec2vec",
        model_name: Optional[str] = None,
    ) -> str:
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(
            run_name=run_name, experiment_id=experiment_id, nested=True
        ) as run:
            params = {
                "n_decimals_for_documents": model.n_decimals,
                "intensity_weighting_power": model.intensity_weighting_power,
                "allowed_missing_percentage": model.allowed_missing_percentage,
                "iter": model.model.epochs,
                "window": model.model.window,
            }
            mlflow.log_params(params)
            run_id = run.info.run_id
            model.set_run_id(run_id)
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name=model_name,
                code_path=[CODE_PATH],
                conda_env=conda_env_path,
            )

            mlflow.log_metric("alpha", model.model.alpha)
            return run_id
