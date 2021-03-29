import mlflow
from mlflow.pyfunc import PythonModel


class Model(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        pass


class ModelRegister:
    def __init__(self, server_uri: str):
        mlflow.set_tracking_uri(server_uri)

    def register_model(
        self,
        model: Model,
        experiment_name: str,
        path: str,
        n_decimals: int,
    ):
        experiment_id = self._get_or_create_experiment_id(experiment_name, path)
        with mlflow.start_run(experiment_id=experiment_id):
            mlflow.log_param("n_decimals_for_documents", n_decimals)
            mlflow.log_param("iter", model.model.iter)
            mlflow.log_param("window", model.model.window)
            mlflow.pyfunc.save_model(
                path=path,
                python_model=model,
                # TODO: maybe add a conda_env
            )
            mlflow.log_metric("alpha", model.model.alpha)

    @staticmethod
    def _get_or_create_experiment_id(experiment_name: str, path: str) -> str:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment:
            return experiment.experiment_id
        else:
            return mlflow.create_experiment(
                experiment_name, artifact_location=f"{path}/artifacts"
            )
