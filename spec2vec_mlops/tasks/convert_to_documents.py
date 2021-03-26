import datetime

from matchms import Spectrum
from prefect import task
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.document_converter import DocumentConverter


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(spectrum: Spectrum, n_decimals: int) -> SpectrumDocument:
    document_converter = DocumentConverter()
    result = document_converter.convert_to_document(spectrum, n_decimals)
    return result
