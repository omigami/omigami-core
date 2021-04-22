import os
from typing import Iterable

import pytest

from spec2vec_mlops.entities.feast_spectrum_document import FeastSpectrumDocument
from spec2vec_mlops.helper_classes.storer_classes import (
    SpectrumIDStorer,
    DocumentStorer,
)
from spec2vec_mlops.tasks.train_model import SpectraIterator, spec2vec_settings

pytestmark = pytest.mark.skipif(
    os.getenv("SKIP_SPARK_TEST", True),
    reason="It can only be run if the Feast docker-compose is up and with Spark",
)


def test_spectra_iterator():
    ids_storer = SpectrumIDStorer("spectrum_ids_info")
    document_storer = DocumentStorer("document_info")
    all_spectrum_ids = ids_storer.read_online()
    spectra_iter = SpectraIterator(all_spectrum_ids, document_storer)

    all_sentences = 0
    all_words = 0
    for sentence in spectra_iter:
        all_sentences += 1
        assert isinstance(sentence, FeastSpectrumDocument)
        for word in sentence:
            all_words += 1
            assert word.startswith("peak@")

    all_words_no_iterator = 0
    for spectrum in document_storer.read_online(all_spectrum_ids):
        all_words_no_iterator += len(spectrum)
    assert all_words == all_words_no_iterator
    assert all_sentences == len(all_spectrum_ids)
    assert isinstance(spectra_iter, Iterable)




