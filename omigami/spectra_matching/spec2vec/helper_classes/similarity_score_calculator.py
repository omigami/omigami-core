from typing import List

import numpy as np
from spec2vec import Spec2Vec
from omigami.spectra_matching.util import cosine_similarity, cosine_similarity_matrix

from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding


class Spec2VecSimilarityScoreCalculator(Spec2Vec):
    """Calculate spec2vec similarity scores between a reference and a query.
    The only difference between Spec2VecSimilarityScoreCalculator and Spec2Vec is that
    Spec2VecSimilarityScoreCalculator methods take as input argument Embedding instead
    of Union[SpectrumDocument, Spectrum]. The Embeddings are being stored at training
    so there is no need to recompute them at every lookup.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pair(self, reference: Spec2VecEmbedding, query: Spec2VecEmbedding) -> float:
        return cosine_similarity(reference.vector, query.vector)

    def matrix(
        self,
        references: List[Spec2VecEmbedding],
        queries: List[Spec2VecEmbedding],
        is_symmetric: bool = False,
    ) -> np.ndarray:

        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(references):
            reference_vectors[index_reference, 0 : self.vector_size] = reference.vector

        n_cols = len(queries)
        if is_symmetric:
            assert np.all(
                references == queries
            ), "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = np.empty((n_cols, self.vector_size), dtype="float")
            for index_query, query in enumerate(queries):
                query_vectors[index_query, 0 : self.vector_size] = query.vector

        spec2vec_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return spec2vec_similarity
