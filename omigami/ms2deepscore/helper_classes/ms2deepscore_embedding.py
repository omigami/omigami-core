from typing import List

import numpy as np
from ms2deepscore import MS2DeepScore
from ms2deepscore.models import SiameseModel
from spec2vec.vector_operations import cosine_similarity, cosine_similarity_matrix
from tqdm import tqdm

from omigami.ms2deepscore.entities.embedding import MS2DeepScoreEmbedding


class MS2DeepScoreSimilarityScoreCalculator(MS2DeepScore):
    """Calculate MS2DeepScore similarity scores between a reference and a query. The
    only difference between MS2DeepScoreSimilarityScoreCalculator and MS2DeepScore is that
    MS2DeepScoreSimilarityScoreCalculator methods take as input argument Embedding instead
    of Spectrum.
    """

    def __init__(self, model: SiameseModel, **kwargs):
        super().__init__(model, **kwargs)

    def pair(self, reference: MS2DeepScoreEmbedding, query: MS2DeepScoreEmbedding) -> float:
        return cosine_similarity(reference.vector[0, :], query.vector[0, :])

    def matrix(
        self,
        references: List[MS2DeepScoreEmbedding],
        queries: List[MS2DeepScoreEmbedding],
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

    def calculate_vectors(self, spectrum_list: List[MS2DeepScoreEmbedding]) -> np.ndarray:
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
            ] = reference.vector
        return reference_vectors
