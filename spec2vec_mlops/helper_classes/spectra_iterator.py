class SpectraIterator:
    """An iterator that yields spectra (lists of str) one by one to the word2vec model for training.
    Reading chunks is not supported by gensim word2vec at the moment.
    """

    def __init__(self, all_spectrum_ids, document_storer):
        self.document_storer = document_storer
        self.all_spectrum_ids = all_spectrum_ids

    def __iter__(self):
        for spectrum_id in self.all_spectrum_ids:
            yield self.document_storer.read_online([spectrum_id])[0]
