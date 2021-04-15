import datetime
from typing import List

from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.storer_classes import DocumentStorer


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def store_documents_task(
    data: List[SpectrumDocument], out_dir: str, feast_core_url: str
):
    storer = DocumentStorer(out_dir, feast_core_url, feature_table_name="document_info")
    storer.store(data)