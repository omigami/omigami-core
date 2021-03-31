import datetime
from typing import List

import numpy as np
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.data_storer import DataStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_embeddings_task(
    data: List[SpectrumDocument],
    embeddings: List[np.ndarray],
    run_id: str,
    out_dir: str,
    feast_core_url: str,
):
    ds = DataStorer(out_dir, feast_core_url)
    ds.store_embeddings(data, embeddings, run_id)
