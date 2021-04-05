from abc import ABC, abstractmethod
from typing import List, Union

import pandas as pd
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops.helper_classes.embedding import Embedding


class BaseStorer(ABC):
    @abstractmethod
    def store(self, data: List[Union[Spectrum, SpectrumDocument, Embedding]]):
        pass

    @abstractmethod
    def _get_data_df(
        self, data: List[Union[Spectrum, SpectrumDocument, Embedding]]
    ) -> pd.DataFrame:
        pass
