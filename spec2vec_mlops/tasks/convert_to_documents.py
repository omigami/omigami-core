import datetime
import logging
from prefect import task
from spec2vec import SpectrumDocument
from spec2vec_mlops import config
from matchms import Spectrum

KEYS = config["gnps_json"]["necessary_keys"].get(list)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


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
