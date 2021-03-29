import datetime

import numpy as np
from gensim.models import Word2Vec
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def make_embeddigns_task(
    model: Word2Vec,
    document: SpectrumDocument,
) -> np.ndarray:
    embedding_maker = EmbeddingMaker()
    embeddings = embedding_maker.make_embeddings(model, document)
    return embeddings
