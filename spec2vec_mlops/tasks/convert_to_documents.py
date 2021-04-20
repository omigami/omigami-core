import datetime
import logging
from typing import List

from prefect import task

from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumStorer,
    DocumentStorer,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(spectrum_ids: List[str], n_decimals: int) -> List[str]:
    chunksize = 100  # capped by document_storer.read_online gRPC message size limit
    ids_chunks = [
        spectrum_ids[i : i + chunksize] for i in range(0, len(spectrum_ids), chunksize)
    ]
    return [
        output_id
        for ids_chunk in ids_chunks
        for output_id in _read_and_convert(ids_chunk, n_decimals)
    ]


def _read_and_convert(spectrum_ids: List[str], n_decimals: int) -> List[str]:
    spectrum_storer = SpectrumStorer("spectrum_info")
    all_spectra = spectrum_storer.read_online(spectrum_ids)
    if all_spectra:
        logger.info(f"Converting {len(all_spectra)} to documents")
    document_converter = DocumentConverter()
    result = [
        document_converter.convert_to_document(spectrum, n_decimals)
        for spectrum in all_spectra
    ]
    document_storer = DocumentStorer("document_info")
    document_storer.store(result)
    return spectrum_ids
