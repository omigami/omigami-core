import datetime
from typing import Union

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.model_register import ModelRegister
from spec2vec_mlops.helper_classes.spec2vec_model import Model


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def register_model_task(
    server_uri: str,
    model: Word2Vec,
    experiment_name: str,
    path: str,
    n_decimals: int,
    intensity_weighting_power: Union[float, int],
    allowed_missing_percentage: Union[float, int],
    conda_env_path: str = None,
) -> str:
    model_register = ModelRegister(server_uri)
    params = {
        "n_decimals_for_documents": n_decimals,
        "intensity_weighting_power": intensity_weighting_power,
        "allowed_missing_percentage": allowed_missing_percentage,
        "iter": model.epochs,
        "window": model.window,
    }
    metrics = {"alpha": model.alpha}
    run_id = model_register.register_model(
        model=Model(model, n_decimals, intensity_weighting_power, allowed_missing_percentage),
        params=params,
        metrics=metrics,
        experiment_name=experiment_name,
        path=path,
        code_to_save=["spec2vec_mlops"],
        conda_env_path=conda_env_path,
    )
    return run_id
