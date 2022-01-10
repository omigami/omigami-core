from omigami.spectra_matching.storage import RedisSpectrumDataGateway
from omigami.spectra_matching.tasks import DeleteEmbeddings


def test_delete_embeddings(ms2deepscore_embeddings_stored):
    dgw = RedisSpectrumDataGateway("ms2deepscore")
    assert len(dgw.read_embeddings("positive")) > 0

    t = DeleteEmbeddings(dgw, "positive")
    t.run()

    assert len(dgw.read_embeddings("positive")) == 0
