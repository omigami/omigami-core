import datetime
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.embedding import Embedding
from spec2vec_mlops.helper_classes.storer_classes import EmbeddingStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_embeddings_task(
    embeddings: List[Embedding],
    run_id: str,
    out_dir: str,
    feast_core_url: str,
):
    storer = EmbeddingStorer(
        out_dir=out_dir,
        feast_core_url=feast_core_url,
        run_id=run_id,
        feature_table_name="embedding_info",
    )
    storer.store(embeddings)
