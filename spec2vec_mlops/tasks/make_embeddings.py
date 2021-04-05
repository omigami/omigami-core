import datetime
from typing import Union, List

from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.helper_classes.embedding import Embedding
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    DocumentStorer,
    EmbeddingStorer,
)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def make_embeddings_task(
    model: Word2Vec,
    run_id: str,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
) -> List[Embedding]:

    # load
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    document_storer = DocumentStorer("document_info")
    all_spectrum_ids = ids_storer.read()
    documents = document_storer.read(all_spectrum_ids)

    # make embeddings
    embedding_maker = EmbeddingMaker()
    embeddings = [
        embedding_maker.make_embedding(
            model,
            document,
            intensity_weighting_power,
            allowed_missing_percentage,
        )
        for document in documents
    ]

    # save
    embedding_storer = EmbeddingStorer("embedding_info", run_id)
    embedding_storer.store(embeddings)
    return embeddings
