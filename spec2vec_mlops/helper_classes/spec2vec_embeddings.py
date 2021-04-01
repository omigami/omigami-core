from typing import List
import numpy as np

from spec2vec import Spec2Vec
from spec2vec.vector_operations import cosine_similarity, cosine_similarity_matrix
from spec2vec_mlops.helper_classes.spec2vec_typing import Embedding


class Spec2VecEmbeddings(Spec2Vec):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def pair(self, reference: Embedding, query: Embedding) -> float:
        return cosine_similarity(reference, query)

    def matrix(
        self,
        references: List[Embedding],
        queries: List[Embedding],
        is_symmetric: bool = False,
    ) -> np.ndarray:

        n_rows = len(references)
        reference_vectors = np.empty((n_rows, self.vector_size), dtype="float")
        for index_reference, reference in enumerate(references):
            reference_vectors[index_reference, 0 : self.vector_size] = reference

        n_cols = len(queries)
        if is_symmetric:
            assert np.all(
                references == queries
            ), "Expected references to be equal to queries for is_symmetric=True"
            query_vectors = reference_vectors
        else:
            query_vectors = np.empty((n_cols, self.vector_size), dtype="float")
            for index_query, query in enumerate(queries):
                query_vectors[index_query, 0 : self.vector_size] = query

        spec2vec_similarity = cosine_similarity_matrix(reference_vectors, query_vectors)
        return spec2vec_similarity
