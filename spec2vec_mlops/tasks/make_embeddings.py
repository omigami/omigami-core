import datetime
from typing import Union, List

import prefect
from gensim.models import Word2Vec
from prefect import task

from spec2vec_mlops.gateways.redis_gateway import RedisDataGateway
from spec2vec_mlops.helper_classes.embedding_maker import EmbeddingMaker


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def make_embeddings_task(
    model: Word2Vec,
    spectrum_ids: List[str],
    run_id: str,
    n_decimals: int,
    intensity_weighting_power: Union[float, int] = 0.5,
    allowed_missing_percentage: Union[float, int] = 5.0,
    chunksize: int = 10,
) -> List[str]:
    logger = prefect.context.get("logger")
    dgw = RedisDataGateway()
    documents = dgw.read_documents(spectrum_ids)

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
    dgw.write_embeddings(embeddings, run_id)
    return spectrum_ids
