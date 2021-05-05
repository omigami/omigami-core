import datetime
from typing import Union

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model
from spec2vec_mlops.tasks.config import DEFAULT_CONFIG

CONDA_ENV_PATH = "requirements/environment.frozen.yaml"


@task(**DEFAULT_CONFIG)
def register_model_task(
    server_uri: str,
    model: Word2Vec,
    experiment_name: str,
    path: str,
    n_decimals: int,
    intensity_weighting_power: Union[float, int],
    allowed_missing_percentage: Union[float, int],
) -> str:
    model_register = ModelRegister(server_uri)
    run_id = model_register.register_model(
        Model(model, n_decimals, intensity_weighting_power, allowed_missing_percentage),
        experiment_name,
        path,
        CONDA_ENV_PATH,
    )
    return run_id
