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

    @staticmethod
    def register_model(
        model: Model,
        experiment_name: str,
        path: str,
        n_decimals: int,
    ):
        mlflow.create_experiment(experiment_name, artifact_location=f"{path}/artifacts")
        mlflow.log_param("n_decimals_for_documents", n_decimals)
        mlflow.log_param("iter", model.model.iter)
        mlflow.log_param("window", model.model.window)
        mlflow.pyfunc.save_model(
            path=path,
            python_model=model,
            # TODO: maybe add a conda_env
        )
        mlflow.log_metric("alpha", model.model.alpha)
