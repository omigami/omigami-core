from omigami.spectra_matching.spec2vec import SPEC2VEC_PROJECT_NAME
from omigami.spectra_matching.spec2vec.entities.embedding import Spec2VecEmbedding
from omigami.spectra_matching.spec2vec.tasks import (
    MakeEmbeddings,
    MakeEmbeddingsParameters,
)
from omigami.spectra_matching.storage import RedisSpectrumDataGateway, FSDataGateway


def test_make_embeddings(
    registered_s2v_model, spec2vec_redis_setup, s3_documents_directory
):
    params = MakeEmbeddingsParameters("positive", 1, 0.5, 15)
    fs_dgw = FSDataGateway()
    document_paths = fs_dgw.list_files(s3_documents_directory)
    spectrum_dgw = RedisSpectrumDataGateway("project")
    t = MakeEmbeddings(spectrum_dgw, fs_dgw, params)

    document_ids = t.run(
        registered_s2v_model["model"].model,
        registered_s2v_model["run_id"],
        document_paths[0],
    )

    assert document_ids
    embeddings = spectrum_dgw.read_embeddings("positive", SPEC2VEC_PROJECT_NAME)
    assert isinstance(embeddings[0], Spec2VecEmbedding)
