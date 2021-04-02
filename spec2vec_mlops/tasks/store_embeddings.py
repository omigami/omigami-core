import datetime
from typing import List

import numpy as np
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.data_storer import EmbeddingStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_embeddings_task(
    data: List[SpectrumDocument],
    embeddings: List[np.ndarray],
    run_id: str,
    out_dir: str,
    feast_core_url: str,
):
    storer = EmbeddingStorer(
        out_dir, feast_core_url, feature_table_name="embedding_info"
    )
    storer.store_embeddings(data, embeddings, run_id)
