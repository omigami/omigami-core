import datetime
from prefect import task
from spec2vec import SpectrumDocument
from matchms import Spectrum


class DocumentConverter:
    def __init__(self):
        pass

    @staticmethod
    def convert_to_document(spectrum: Spectrum, n_decimals: int) -> SpectrumDocument:
        return SpectrumDocument(spectrum, n_decimals=n_decimals)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(spectrum: Spectrum, n_decimals: int) -> SpectrumDocument:
    document_converter = DocumentConverter()
    result = document_converter.convert_to_document(spectrum, n_decimals)
    return result
