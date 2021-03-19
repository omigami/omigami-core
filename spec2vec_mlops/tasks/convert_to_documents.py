import datetime
import logging
from typing import List
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
    def convert_to_documents(
        spectra: List[Spectrum], n_decimals: int
    ) -> List[SpectrumDocument]:
        return [SpectrumDocument(s, n_decimals=n_decimals) for s in spectra]


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(
    spectra: List[Spectrum], n_decimals: int
) -> List[SpectrumDocument]:
    document_converter = DocumentConverter()
    results = document_converter.convert_to_documents(spectra, n_decimals)
    return results
