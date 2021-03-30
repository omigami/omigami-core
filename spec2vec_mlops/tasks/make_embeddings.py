import datetime
from typing import Union

import numpy as np
from gensim.models import Word2Vec
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def make_embeddings_task(
    model: Word2Vec,
    document: SpectrumDocument,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> np.ndarray:
    embedding_maker = EmbeddingMaker()
    embeddings = embedding_maker.make_embedding(
        model,
        document,
        intensity_weighting_power,
        allowed_missing_percentage,
    )
    return embeddings
