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
    spectrum_storer = SpectrumStorer("spectrum_info")
    all_spectra = spectrum_storer.read_online(spectrum_ids)
    if all_spectra:
        logger.info(f"Using previously downloaded data")
    document_converter = DocumentConverter()
    result = [
        document_converter.convert_to_document(spectrum, n_decimals)
        for spectrum in all_spectra
    ]
    document_storer = DocumentStorer("document_info")
    document_storer.store(result)
    return spectrum_ids
