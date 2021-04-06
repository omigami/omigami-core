from typing import Dict, List, Any

from spec2vec import Document


class FeastSpectrumDocument(Document):
    """Document as output from data stored in Feast."""

    def __init__(self, feast_data: Dict[str, List[Any]]):
        super().__init__(obj=feast_data)
        self.weights = feast_data["weights"]
        self.losses = feast_data["losses"]
        self.metadata = feast_data["metadata"]
        self.n_decimals = feast_data["n_decimals"]

    def _make_words(self):
        """Create word from Feast data."""
        self.words = self._obj["words"]
        return self