from typing import List
from tqdm import tqdm
import numpy as np

from ms2deepscore import MS2DeepScore, BinnedSpectrum
from spec2vec.vector_operations import cosine_similarity, cosine_similarity_matrix


class MS2DeepScoreBinnedSpectrum(MS2DeepScore):
    """Calculate MS2DeepScore similarity scores between a reference and a query. The
    only difference between MS2DeepScoreBinnedSpectrum and MS2DeepScore is that
    MS2DeepScoreBinnedSpectrum methods take as input argument BinnedSpectrum instead
    of Spectrum.
    """

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)

    def pair(self, reference: BinnedSpectrum, query: BinnedSpectrum) -> float:
        reference_vector = self.model.base.predict(self._create_input_vector(reference))
        query_vector = self.model.base.predict(self._create_input_vector(query))

        return cosine_similarity(reference_vector[0, :], query_vector[0, :])

    def matrix(
        self,
        references: List[BinnedSpectrum],
        queries: List[BinnedSpectrum],
        is_symmetric: bool = False,
    ) -> np.ndarray:

        reference_vectors = self.calculate_vectors(references)
        if is_symmetric:
            assert np.all(
                references == queries
            ), "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = self.calculate_vectors(queries)

        ms2ds_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return ms2ds_similarity

    def calculate_vectors(self, spectrum_list: List[BinnedSpectrum]) -> np.ndarray:
        n_rows = len(spectrum_list)
        reference_vectors = np.empty((n_rows, self.output_vector_dim), dtype="float")
        for index_reference, reference in enumerate(
            tqdm(
                spectrum_list,
                desc="Calculating vectors of reference spectrums",
                disable=(not self.progress_bar),
            )
        ):
            reference_vectors[
                index_reference, 0 : self.output_vector_dim
            ] = self.model.base.predict(self._create_input_vector(reference))
        return reference_vectors
