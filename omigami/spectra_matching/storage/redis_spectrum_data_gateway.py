from __future__ import annotations

import pickle
from logging import Logger, getLogger
from typing import List, Iterable, Set

from matchms import Spectrum

from omigami.config import (
    SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
    SPECTRUM_HASHES,
    EMBEDDING_HASHES,
)
from omigami.spectra_matching.entities.embedding import Embedding
from omigami.spectra_matching.storage import RedisDataGateway


log = getLogger(__name__)

class RedisSpectrumDataGateway(RedisDataGateway):
    """Data gateway for Redis storage."""

    def write_raw_spectra(self, spectra: List[Spectrum]):
        """Writes a list of raw spectra to the redis database using the spectrum_id as the key.

        Parameters
        ----------
        spectra: List[Spectrum]
            List containing objects the class matchms.Spectrum.
        """
        self._init_client()

        for spectrum in spectra:
            self.client.zadd(
                SPECTRUM_ID_PRECURSOR_MZ_SORTED_SET,
                {spectrum.metadata["spectrum_id"]: spectrum.metadata["precursor_mz"]},
            )
            self.client.hset(
                SPECTRUM_HASHES,
                spectrum.metadata["spectrum_id"],
                pickle.dumps(spectrum),
            )

    def list_spectrum_ids(self) -> List[str]:
        """List the spectrum ids of all spectra on the redis database."""
        self._init_client()
        return [id_.decode() for id_ in self.client.hkeys(SPECTRUM_HASHES)]

    def list_existing_spectra(self, spectrum_ids: List[str]) -> Set[str]:
        self._init_client()
        existing = set(self.client.hkeys(SPECTRUM_HASHES))
        return {sp_id for sp_id in spectrum_ids if sp_id.encode() in existing}

    def read_spectra(self, spectrum_ids: Iterable[str] = None) -> List[Spectrum]:
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
        return spectra

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
        log.info("Finished getting spectrum_ids_within_range")
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

    def _list_missing_spectrum_ids(
        self, hash_name: str, spectrum_ids: List[str]
    ) -> List[str]:
        self._init_client()
        return [id for id in spectrum_ids if not self.client.hexists(hash_name, id)]

    def write_embeddings(
        self,
        embeddings: List[Embedding],
        ion_mode: str,
        logger: Logger = None,
    ):
        """Write embeddings data on the redis database."""
        self._init_client()
        hash_key = self._format_redis_key(hashes=EMBEDDING_HASHES, ion_mode=ion_mode)
        if logger:
            logger.debug(
                f"Saving {len(embeddings)} embeddings to the client {self.client}"
                f" on hash '{hash_key}'."
            )
        pipe = self.client.pipeline()
        for embedding in embeddings:
            pipe.hset(
                hash_key,
                embedding.spectrum_id,
                pickle.dumps(embedding),
            )
        pipe.execute()

    def read_embeddings(
        self, ion_mode: str, spectrum_ids: List[str] = None
    ) -> List[Embedding]:
        """Read the embeddings from spectra IDs.
        Return a list of Embedding objects."""
        self._init_client()
        return self._read_hashes(
            self._format_redis_key(hashes=EMBEDDING_HASHES, ion_mode=ion_mode),
            spectrum_ids,
        )

    def delete_embeddings(self, ion_mode: str):
        """Deletes embeddings for a project + ion mode combination."""
        self._init_client()
        hash_key = self._format_redis_key(EMBEDDING_HASHES, ion_mode)
        self.client.delete(hash_key)
