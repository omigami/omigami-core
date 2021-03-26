import datetime

import mlflow
from gensim.models import Word2Vec
from mlflow.pyfunc import PythonModel
from prefect import task


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


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def register_model_task(
    model: Word2Vec,
    path: str,
    n_decimals: int,
):
    model_register = ModelRegister()
    model_register.register_model(Model(model), path, n_decimals)
