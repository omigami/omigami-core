import datetime
import logging
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.storer_classes import SpectrumIDStorer

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def load_spectrum_ids_task(chunksize: int = 1000) -> List[List[str]]:
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    all_spectrum_ids = ids_storer.read_online()
    logger.info(f"Read {len(all_spectrum_ids)} spectra from online store")
    all_spectrum_ids_chunks = [
        all_spectrum_ids[i : i + chunksize]
        for i in range(0, len(all_spectrum_ids), chunksize)
    ]
    return all_spectrum_ids_chunks
