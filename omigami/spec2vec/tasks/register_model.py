from dataclasses import dataclass
from typing import Union, Dict

import mlflow
from gensim.models import Word2Vec
from omigami.model_register import MLFlowModelRegister
from omigami.spec2vec.predictor import Spec2VecPredictor
from omigami.utils import merge_prefect_task_configs
from prefect import Task

CONDA_ENV_PATH = "./requirements/environment.frozen.yaml"


@dataclass
class RegisterModelParameters:
    experiment_name: str
    mlflow_output_path: str
    server_uri: str
    n_decimals: int
    intensity_weighting_power: Union[float, int]
    allowed_missing_percentage: Union[float, int]


class RegisterModel(Task):
    def __init__(
        self,
        parameters: RegisterModelParameters,
        **kwargs,
    ):
        self._experiment_name = parameters.experiment_name
        self._path = parameters.mlflow_output_path
        self._n_decimals = parameters.n_decimals
        self._intensity_weighting_power = parameters.intensity_weighting_power
        self._allowed_missing_percentage = parameters.allowed_missing_percentage
        self._server_uri = parameters.server_uri
        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self, model: Word2Vec = None) -> Dict[str, str]:
        self.logger.info(
            f"Registering model to {self._server_uri} on URI: {self._path}."
        )
        model_register = ModelRegister(self._server_uri)
        run_id = model_register.register_model(
            Spec2VecPredictor(
                model,
                self._n_decimals,
                self._intensity_weighting_power,
                self._allowed_missing_percentage,
            ),
            self._experiment_name,
            self._path,
            CONDA_ENV_PATH,
        )
        run = mlflow.get_run(run_id)
        self.logger.info(f"{run.info}")

        model_uri = f"{run.info.artifact_uri}/model/"
        return {"model_uri": model_uri, "run_id": run_id}


class ModelRegister(MLFlowModelRegister):
    def register_model(
        self,
        model: Spec2VecPredictor,
        experiment_name: str,
        output_path: str = None,
        conda_env_path: str = None,
    ) -> str:
        experiment_id = self._get_or_create_experiment_id(experiment_name, output_path)
        with mlflow.start_run(experiment_id=experiment_id, nested=True) as run:
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
            self.log_model(
                model,
                experiment_name,
                output_path=output_path,
                code_path=["omigami"],
                conda_env_path=conda_env_path,
            )

            mlflow.log_metric("alpha", model.model.alpha)
            return run_id
