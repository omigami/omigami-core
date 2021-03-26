import mlflow
from mlflow.pyfunc import PythonModel


class Model(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        pass


class ModelRegister:
    @staticmethod
    def register_model(
        model: Model,
        path: str,
        n_decimals: int,
    ):
        mlflow.log_param("n_decimals_for_documents", n_decimals)
        mlflow.pyfunc.save_model(
            path=path,
            python_model=model,
            # TODO: maybe add a conda_env
        )
        mlflow.log_metric("iter", model.model.iter)
        mlflow.log_metric("window", model.model.window)
        mlflow.log_metric("alpha", model.model.alpha)
