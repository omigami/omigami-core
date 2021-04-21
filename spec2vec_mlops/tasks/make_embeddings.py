import datetime
from typing import Union, List

import prefect
from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.storer_classes import (
    DocumentStorer,
    EmbeddingStorer,
)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def make_embeddings_task(
    model: Word2Vec,
    spectrum_ids: List[str],
    run_id: str,
    n_decimals: int,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> List[str]:
    logger = prefect.context.get("logger")
    document_storer = DocumentStorer("document_info")
    chunksize = 10  # capped by storer.read_online gRPC message size limit
    ids_chunks = [
        spectrum_ids[i : i + chunksize] for i in range(0, len(spectrum_ids), chunksize)
    ]
    documents = [
        doc for chunk in ids_chunks for doc in document_storer.read_online(chunk)
    ]

    logger.info(f"Make {len(documents)} embeddings")
    embedding_maker = EmbeddingMaker(n_decimals=n_decimals)
    embeddings = [
        embedding_maker.make_embedding(
            model,
            document,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
        for document in documents
    ]
    embedding_storer = EmbeddingStorer("embedding_info", run_id)
    embedding_storer.store(embeddings)
    return spectrum_ids
