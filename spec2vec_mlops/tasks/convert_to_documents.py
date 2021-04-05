import datetime

from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    SpectrumStorer,
)
from spec2vec_mlops.helper_classes.document_converter import DocumentConverter


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(feast_core_url: str, n_decimals: int) -> SpectrumDocument:
    spectrum_ids_storer = SpectrumIDStorer(feast_core_url)
    spectrum_storer = SpectrumStorer(feast_core_url)
    all_spectrum_ids = spectrum_ids_storer.read()
    spectrum = spectrum_storer.read(all_spectrum_ids)

    document_converter = DocumentConverter()
    result = document_converter.convert_to_document(spectrum, n_decimals)
    return result
