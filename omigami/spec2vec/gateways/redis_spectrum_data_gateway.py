from __future__ import annotations

import os
import pickle
from logging import Logger
from typing import List, Iterable, Dict, Set

import redis
from matchms import Spectrum
from spec2vec import SpectrumDocument

from omigami.spec2vec.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    REDIS_HOST,
    REDIS_DB,
)
from omigami.gateways.data_gateway import SpectrumDataGateway
from omigami.spec2vec.entities.embedding import Embedding
from omigami.spec2vec.entities.spectrum_document import SpectrumDocumentData


# when running locally, those should be set in pycharm/shell env
# when running on the cluster, they will be gotten from the seldon env,
# which was defined during deployment by the 'dataset_name' param
REDIS_HOST = str(os.getenv(REDIS_HOST))
REDIS_DB = str(os.getenv(REDIS_DB))

client = None


def get_redis_client():
    global client
    if client is None:
        client = redis.StrictRedis(host=REDIS_HOST, db=REDIS_DB)
    return client


class RedisSpectrumDataGateway(SpectrumDataGateway):
    """Data gateway for Redis storage."""

    def __init__(self):
        # We initialize it with None so we can pickle this gateway when deploying the flow
        self.client = None

    def _init_client(self):
        if self.client is None:
            self.client = get_redis_client()

    def list_spectrum_ids(self) -> List[str]:
        """List the spectrum ids of all spectra on the redis database."""
        self._init_client()
        return [id_.decode() for id_ in self.client.hkeys(SPECTRUM_HASHES)]

    def list_existing_spectra(self, spectrum_ids: List[str]) -> Set[str]:
        self._init_client()
        existing = set(self.client.hkeys(SPECTRUM_HASHES))
        return {sp_id for sp_id in spectrum_ids if sp_id.encode() in existing}

    def read_spectra(self, spectrum_ids: Iterable[str] = None) -> Dict[str, Spectrum]:
        """
        Read the spectra information from spectra IDs.
        Return a dict of Spectrum objects.

        Parameters
        ----------
        spectrum_ids:
            List of spectrum ids to be read from the database

        Returns
        -------
        spectra;
            Dictionary of `spectrum_id: spectrum`

        """
        self._init_client()
        spectra = self._read_hashes(SPECTRUM_HASHES, spectrum_ids)
        return {spectrum.metadata["spectrum_id"]: spectrum for spectrum in spectra}

    def get_spectrum_ids_within_range(
        self, min_mz: float = 0, max_mz: float = -1
    ) -> List[str]:
        """Get the spectrum IDs of spectra stored on redis that have a Precursor_MZ
        within the given range. Return a list spectrum IDs."""
        self._init_client()
        spectrum_ids_within_range = [
            id_.decode()
            for id_ in self.client.zrangebyscore(
                SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET, min_mz, max_mz
            )
        ]
        return spectrum_ids_within_range

    def _read_hashes(self, hash_name: str, spectrum_ids: List[str] = None) -> List:
        if spectrum_ids:
            spectra = self.client.hmget(hash_name, spectrum_ids)
            loaded_spectra = [pickle.loads(s) for s in spectra if s]
            return loaded_spectra
        else:
            return [pickle.loads(e) for e in self.client.hgetall(hash_name).values()]

    def delete_spectra(self, spectrum_ids: List[str]):
        # Just used on tests atm. No abstract method.
        self._init_client()
        _ = [self.client.hdel(SPECTRUM_HASHES, id_.encode()) for id_ in spectrum_ids]

    def write_spectrum_documents(self, spectra_data: List[SpectrumDocumentData]):
        pass

    def write_raw_spectra(self, spectra):
        self._init_client()

        for spectrum in spectra:
            # spectrum_info = spectrum.spectrum
            self.client.hset(
                SPECTRUM_HASHES, spectrum["spectrum_id"], pickle.dumps(spectrum)
            )

    def write_embeddings(
        self, embeddings: List[Embedding], run_id: str, logger: Logger = None
    ):
        pass

    def read_documents(self, spectrum_ids: List[str] = None) -> List[SpectrumDocument]:
        pass

    def read_documents_iter(self, spectrum_ids: List[str] = None) -> Iterable:
        pass


class RedisHashesIterator:
    """An iterator that yields Redis object one by one to the word2vec model for training.
    Reading chunks is not supported by gensim word2vec at the moment.
    """

    def __init__(
        self,
        dgw: RedisSpectrumDataGateway,
        hash_name: str,
        spectrum_ids: List[str] = None,
    ):
        self.dgw = dgw
        self.hash_name = hash_name
        self.spectrum_ids = (
            [s.encode() for s in spectrum_ids]
            if spectrum_ids
            else dgw.client.hkeys(hash_name)
        )

    def __iter__(self):
        for spectrum_id in self.spectrum_ids:
            data = self.dgw.client.hget(self.hash_name, spectrum_id)
            if not data:
                raise RuntimeError(
                    f"There is no document for spectrum id {spectrum_id}."
                )
            yield pickle.loads(data)