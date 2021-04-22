from typing import Iterable

from spec2vec.utils import TrainingProgressLogger

from spec2vec_mlops.entities.feast_spectrum_document import FeastSpectrumDocument
from spec2vec_mlops.tasks.train_model import MyCorpus, spec2vec_settings


def test_iterable_corpus():
    corpus = MyCorpus()
    all_sentences = 0
    all_words = 0
    for sentence in corpus:
        all_sentences += 1
        assert isinstance(sentence, FeastSpectrumDocument)
        for word in sentence:
            all_words += 1
            assert word.startswith("peak@")

    assert isinstance(corpus, Iterable)
    assert all_sentences == 100
    assert all_words == 33488


def test_spec2vec_settings():
    iterations = 5
    callbacks, settings = spec2vec_settings(iterations=iterations)
    assert isinstance(callbacks[0], TrainingProgressLogger)
    assert settings["iter"] == iterations
