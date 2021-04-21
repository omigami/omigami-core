import datetime
from typing import List

import prefect
from prefect import task

from spec2vec_mlops.helper_classes.document_converter import DocumentConverter
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumStorer,
    DocumentStorer,
)


@task(max_retries=3, retry_delay=datetime.timedelta(seconds=10))
def convert_to_documents_task(spectrum_ids: List[str], n_decimals: int) -> List[str]:
    logger = prefect.context.get("logger")
    spectrum_storer = SpectrumStorer("spectrum_info")
    chunksize = 10  # capped by storer.read_online gRPC message size limit
    ids_chunks = [
        spectrum_ids[i : i + chunksize] for i in range(0, len(spectrum_ids), chunksize)
    ]
    all_spectra = [
        spectrum
        for chunk in ids_chunks
        for spectrum in spectrum_storer.read_online(chunk)
    ]

    logger.info(f"Converting {len(all_spectra)} to documents")
    document_converter = DocumentConverter()
    result = [
        document_converter.convert_to_document(spectrum, n_decimals)
        for spectrum in all_spectra
    ]
    document_storer = DocumentStorer("document_info")
    document_storer.store(result)
    return spectrum_ids
