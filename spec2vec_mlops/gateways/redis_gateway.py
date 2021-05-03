import os
import pickle
from typing import List, Iterable

import redis
from matchms import Spectrum
from spec2vec import SpectrumDocument

from spec2vec_mlops import config
from spec2vec_mlops.entities.spectrum_document import SpectrumDocumentData
from spec2vec_mlops.entities.embedding import Embedding

HOST = os.getenv("REDIS_HOST", config["redis"]["host"])
DB = os.getenv("REDIS_DB", config["redis"]["db"])
SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET = config["redis"]["spectrum_id_sorted_set"]
SPECTRUM_HASHES = config["redis"]["spectrum_hashes"]
DOCUMENT_HASHES = config["redis"]["document_hashes"]
EMBEDDING_HASHES = config["redis"]["embedding_hashes"]


class RedisDataGateway:
    """Data gateway for Redis storage."""

    def __init__(self):
        self.client = redis.StrictRedis(host=HOST, db=DB)

    def write_spectrum_documents(self, spectra_data: List[SpectrumDocumentData]):
        """Write spectrum and document to Redis. Also write a sorted set of spectrum_ids."""
        pipe = self.client.pipeline()
        for spectrum in spectra_data:
            spectrum_info = spectrum.spectrum
            document = spectrum.document
            if spectrum_info and document:
                pipe.zadd(
                    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
                    {spectrum.spectrum_id: spectrum.precursor_mz},
                )
                pipe.hset(
                    SPECTRUM_HASHES, spectrum.spectrum_id, pickle.dumps(spectrum_info)
                )
                pipe.hset(DOCUMENT_HASHES, spectrum.spectrum_id, pickle.dumps(document))
        pipe.execute()

    def write_embeddings(self, embeddings: List[Embedding], run_id: str):
        """Write embeddings to Redis."""
        pipe = self.client.pipeline()
        for embedding in embeddings:
            pipe.hset(
                f"{EMBEDDING_HASHES}_{run_id}",
                embedding.spectrum_id,
                pickle.dumps(embedding),
            )
        pipe.execute()

    def list_spectra_not_exist(self, spectrum_ids: List[str]):
        """Check whether spectra exist on Redis.
        Return a list of IDs that do not exist.
        """
        return self._list_spectrum_ids_not_exist(SPECTRUM_HASHES, spectrum_ids)

    def list_documents_not_exist(self, spectrum_ids: List[str]):
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        return self._list_spectrum_ids_not_exist(DOCUMENT_HASHES, spectrum_ids)

    def read_spectra(self, spectrum_ids: List[str] = None) -> List[Spectrum]:
        return self._read_hashes(SPECTRUM_HASHES, spectrum_ids)

    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        return self._read_hashes(DOCUMENT_HASHES, spectrum_ids)

    def read_embeddings(
        self, run_id: str, spectrum_ids: List[str] = None
    ) -> List[Embedding]:
        return self._read_hashes(f"{EMBEDDING_HASHES}_{run_id}", spectrum_ids)

    def read_embeddings_within_range(
        self, run_id: str, min_mz: int = 0, max_mz: int = -1
    ) -> List[Embedding]:
        spectra_ids_within_range = self._read_spectra_ids_within_range(
            SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, min_mz, max_mz
        )
        return self.read_embeddings(run_id, spectra_ids_within_range)

    def read_documents_iter(self) -> Iterable:
        return RedisHashesIterator(self, DOCUMENT_HASHES)

    def _read_hashes(self, hash_name: str, spectrum_ids: List[str] = None):
        if spectrum_ids:
            return [
                pickle.loads(self.client.hget(hash_name, id)) for id in spectrum_ids
            ]
        else:
            return [pickle.loads(e) for e in self.client.hgetall(hash_name).values()]

    def _list_spectrum_ids_not_exist(
        self, hash_name: str, spectrum_ids: List[str]
    ) -> List[str]:
        return [id for id in spectrum_ids if not self.client.hexists(hash_name, id)]

    def _read_spectra_ids_within_range(self, hash_name: str, min_mz: int, max_mz: int):
        return self.client.zrangebyscore(hash_name, min_mz, max_mz)


class RedisHashesIterator:
    """An iterator that yields Redis object one by one to the word2vec model for training.
    Reading chunks is not supported by gensim word2vec at the moment.
    """

    def __init__(self, dgw, hash_name):
        self.dgw = dgw
        self.hash_name = hash_name
        self.spectra_ids = dgw.client.hkeys(hash_name)

    def __iter__(self):
        for spectrum_id in self.spectra_ids:
            data = self.dgw.client.hmget(self.hash_name, spectrum_id)[0]
            yield pickle.loads(data)
