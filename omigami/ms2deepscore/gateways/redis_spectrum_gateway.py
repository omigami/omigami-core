from __future__ import annotations

import pickle
from logging import Logger
from typing import List

from ms2deepscore import BinnedSpectrum

from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES


class MS2DeepScoreRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for Redis storage."""

    def list_binned_spectra_not_exist(self, spectrum_ids: List[str]) -> List[str]:
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        self._init_client()
        return self._list_spectrum_ids_not_exist(BINNED_SPECTRUM_HASHES, spectrum_ids)

    def write_binned_spectra(
        self, binned_spectra: List[BinnedSpectrum], logger: Logger = None
    ):
        """Write binned spectra data on the redis database."""
        self._init_client()
        if logger:
            logger.debug(
                f"Saving {len(binned_spectra)} binned spectra to the client {self.client}"
                f" on hash '{BINNED_SPECTRUM_HASHES}'."
            )
        pipe = self.client.pipeline()
        for spectrum in binned_spectra:
            pipe.hset(
                f"{BINNED_SPECTRUM_HASHES}",
                spectrum.metadata["spectrum_id"],
                pickle.dumps(spectrum),
            )
        pipe.execute()
