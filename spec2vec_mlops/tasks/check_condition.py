from typing import List, Any, Optional

from gensim.models import Word2Vec
from prefect import task


@task
def check_condition(inputs: Optional[List[Any]]):
    return True if inputs and len(inputs) > 0 else False


@task
def check_model_condition(model: Optional[Word2Vec]):
    return True if model else False
