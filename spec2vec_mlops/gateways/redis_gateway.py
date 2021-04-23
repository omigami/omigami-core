import os
import pickle
from typing import List, Generator

import redis
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.entities.embedding import Embedding

HOST = os.getenv("REDIS_HOST", config["redis"]["host"])
DB = config["redis"]["db"]
SPECTRUM_SET_NAME = config["redis"]["spectrum_set_name"]


class RedisDataGateway:
    """Data gateway for Redis storage."""

    def __init__(self):
        self.client = redis.StrictRedis(host=HOST, db=DB)

    def write_spectrum_documents(self, spectra_data: List[SpectrumDocumentData]):
        pipe = self.client.pipeline()
        for spectrum in spectra_data:
            pipe.set(spectrum.spectrum_id, pickle.dumps(spectrum.spectrum))
            pipe.zadd(
                SPECTRUM_SET_NAME,
                {pickle.dumps(spectrum.document): spectrum.precursor_mz},
            )
        pipe.execute()

    def write_embeddings(self, embeddings: List[Embedding]):
        pass
        # pipe = self.client.pipeline()
        # for embedding in embeddings:
        #     pipe.set(embedding.spectrum_id, pickle.dumps(embedding))
        # pipe.execute()

    def read_documents_iter(self) -> Generator:
        return self.client.zscan_iter(SPECTRUM_SET_NAME)

    def read_documents(self, mz_min: float, mz_max: float) -> List[SpectrumDocument]:
        docs = self.client.zrange(SPECTRUM_SET_NAME, mz_min, mz_max)
        return [pickle.loads(d) for d in docs]

    def read_spectra(self, spectrum_ids: List[str]) -> List[Spectrum]:
        return [self.client.get(id) for id in spectrum_ids]
