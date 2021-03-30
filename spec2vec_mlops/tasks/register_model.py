import datetime

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.model_register import Model, ModelRegister


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def register_model_task(
    server_uri: str,
    model: Word2Vec,
    experiment_name: str,
    path: str,
    n_decimals: int,
    conda_env_path: str = None,
):
    model_register = ModelRegister(server_uri)
    model_register.register_model(
        Model(model), experiment_name, path, n_decimals, conda_env_path
    )
