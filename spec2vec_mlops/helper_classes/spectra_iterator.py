class SpectraIterator:
    """An iterator that yields spectra (lists of str)."""
    def __init__(self, all_spectrum_ids, document_storer):
        self.document_storer = document_storer
        self.all_spectrum_ids = all_spectrum_ids

    def __iter__(self):
        for spectrum_id in self.all_spectrum_ids:
            yield self.document_storer.read_online([spectrum_id])[0]
