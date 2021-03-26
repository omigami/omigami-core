import datetime

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.model_register import Model, ModelRegister


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def register_model_task(
    model: Word2Vec,
    path: str,
    n_decimals: int,
):
    model_register = ModelRegister()
    model_register.register_model(Model(model), path, n_decimals)
