from dataclasses import dataclass
from typing import Union, Dict, Tuple

import gensim
import mlflow
import prefect
from gensim.models import Word2Vec
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from prefect import Task
from spec2vec.model_building import (
    set_spec2vec_defaults,
    learning_rates_to_gensim_style,
)
import mlflow.pyfunc

from omigami.data_gateway import SpectrumDataGateway
from omigami.predictor import Predictor
from omigami.tasks.config import merge_configs

CONDA_ENV_PATH = "./requirements/environment.frozen.yaml"


@dataclass
class TrainModelParameters:
    # training
    epochs: int
    window: int
    n_decimals: int
    intensity_weighting_power: Union[float, int]
    allowed_missing_percentage: Union[float, int]

    # mlflow
    experiment_name: str
    model_output_path: str
    server_uri: str
    use_latest: bool = False


class TrainModel(Task):
    def __init__(
        self, params: TrainModelParameters, spectrum_dgw: SpectrumDataGateway, **kwargs
    ):
        self._params = params
        self._spectrum_dgw = spectrum_dgw
        self._client = MlflowClient(params.server_uri)

        mlflow.set_tracking_uri(params.server_uri)
        config = merge_configs(kwargs)
        super().__init__(**config, trigger=prefect.triggers.all_successful)

    def run(self, model: Word2Vec = None) -> Dict[str, str]:
        if self._params.use_latest:
            registry = self._get_latest_model()
            self.logger.info(f"Using existing model: {registry}")
            return registry

        experiment_id = self._get_or_create_experiment_id()
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            self.logger.info(f"Created run {run_id}.")

            model = self._train_model()
            self._register_model(
                run_id=run_id,
                model=model,
            )
            self.logger.info(f"Finished training the model and saving to MLFlow.")

        return {"model_uri": f"{run.info.artifact_uri}/model/", "run_id": run_id}

    def _train_model(self):
        self.logger.info(
            f"Started training model for {self._params.epochs}"
            f" iterations and {self._params.window} window size."
        )
        documents = self._spectrum_dgw.read_documents_iter()
        callbacks, settings = self._create_spec2vec_settings()
        model = gensim.models.Word2Vec(
            sentences=documents, callbacks=callbacks, **settings
        )
        return model

    def _get_latest_model(self) -> Dict[str, str]:
        model_versions = self._client.search_model_versions(
            f"name='{self._params.experiment_name}'"
        )
        latest_version = sorted(model_versions, key=lambda x: x.version)[-1]
        registry = {"model_uri": latest_version.source, "run_id": latest_version.run_id}
        return registry

    def _register_model(self, run_id: str, model: Word2Vec):
        self.logger.info(
            f"Registering model to {self._params.server_uri} on URI: "
            f"{self._params.model_output_path}."
        )
        model = Predictor(
            model,
            self._params.n_decimals,
            self._params.intensity_weighting_power,
            self._params.allowed_missing_percentage,
            run_id=run_id,
        )
        params = {
            "n_decimals_for_documents": self._params.n_decimals,
            "intensity_weighting_power": self._params.intensity_weighting_power,
            "allowed_missing_percentage": self._params.allowed_missing_percentage,
            "iter": self._params.epochs,
            "window": self._params.window,
        }
        mlflow.log_params(params)

        try:
            mlflow.pyfunc.log_model(
                "model",
                python_model=model,
                registered_model_name=self._params.experiment_name,
                conda_env=CONDA_ENV_PATH,
                code_path=[
                    "omigami",
                ],
            )
        # This is need to run the flow locally. mlflow.pyfunc.log_model is not
        # supported without a database.
        except MlflowException:
            mlflow.pyfunc.save_model(
                f"{self._params.model_output_path}/model",
                python_model=model,
                conda_env=CONDA_ENV_PATH,
                code_path=[
                    "omigami",
                ],
            )
        mlflow.log_metric("alpha", model.model.alpha)

    def _get_or_create_experiment_id(self) -> str:
        experiment = mlflow.get_experiment_by_name(self._params.experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(
                self._params.experiment_name,
                artifact_location=self._params.model_output_path,
            )

    def _create_spec2vec_settings(self) -> Tuple[list, dict]:
        settings = set_spec2vec_defaults(
            epochs=self._params.epochs, window=self._params.window
        )
        settings = learning_rates_to_gensim_style(
            num_of_epochs=self._params.epochs, **settings
        )

        callbacks = []

        # TODO: implement our own logger because this one doesn't work and doesn't log
        # TODO: progress in prefect
        # training_progress_logger = TrainingProgressLogger(num_of_epochs)
        # callbacks.append(training_progress_logger)

        return callbacks, settings
