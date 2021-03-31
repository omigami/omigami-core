import mlflow
from mlflow.exceptions import MlflowException

from spec2vec_mlops.helper_classes.spec2vec_model import Model


class ModelRegister:
    def __init__(self, server_uri: str):
        mlflow.set_tracking_uri(server_uri)

    def register_model(
        self,
        model: Model,
        experiment_name: str,
        path: str,
        conda_env_path: str = None,
    ) -> str:
        experiment_id = self._get_or_create_experiment_id(experiment_name, path)
        with mlflow.start_run(experiment_id=experiment_id) as run:
            params = {
                "n_decimals_for_documents": model.n_decimals,
                "intensity_weighting_power": model.intensity_weighting_power,
                "allowed_missing_percentage": model.allowed_missing_percentage,
                "iter": model.model.iter,
                "window": model.model.window,
            }
            mlflow.log_params(params)
            try:
                mlflow.pyfunc.log_model(
                    "model",
                    python_model=model,
                    registered_model_name=experiment_name,
                    conda_env=conda_env_path,
                )
            # This is need to run the flow locally. mlflow.pyfunc.log_model is not supported without a database
            except MlflowException:
                mlflow.pyfunc.save_model(
                    f"{path}/model",
                    python_model=model,
                    conda_env=conda_env_path,
                )
            mlflow.log_metric("alpha", model.model.alpha)
            return run.info.run_id

    @staticmethod
    def _get_or_create_experiment_id(experiment_name: str, path: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(experiment_name, artifact_location=path)
