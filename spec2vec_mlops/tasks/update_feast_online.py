import datetime
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.storer_classes import SpectrumStorer, \
    DocumentStorer, EmbeddingStorer, SpectrumIDStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def update_feast_online_task(storer_class: str, spectrum_ids_chunks: List[List[str]], **kwargs) -> List[List[str]]:
    if storer_class == "spectrum":
        storer = SpectrumStorer("spectrum_info")
    elif storer_class == "document":
        storer = DocumentStorer("document_info")
    elif storer_class == "embedding":
        storer = EmbeddingStorer("embedding_info", kwargs["run_id"])
    else:
        storer = SpectrumIDStorer("spectrum_ids_info")
    storer.store_online()
    return spectrum_ids_chunks
