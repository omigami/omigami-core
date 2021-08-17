from __future__ import annotations

import pickle
from logging import Logger
from typing import List

from ms2deepscore import BinnedSpectrum
from omigami.config import IonModes
from omigami.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.ms2deepscore.config import BINNED_SPECTRUM_HASHES

from omigami.ms2deepscore.config import PROJECT_NAME


class MS2DeepScoreRedisSpectrumDataGateway(RedisSpectrumDataGateway):
    """Data gateway for Redis storage."""

    def __init__(self, project=PROJECT_NAME):

        super().__init__(project)

    def list_missing_binned_spectra(
        self, spectrum_ids: List[str], ion_mode: IonModes
    ) -> List[str]:
        """Check whether document exist on Redis.
        Return a list of IDs that do not exist.
        """
        self._init_client()
        return self._list_missing_spectrum_ids(
            f"{BINNED_SPECTRUM_HASHES}_{ion_mode}", spectrum_ids
        )

    def write_binned_spectra(
        self,
        binned_spectra: List[BinnedSpectrum],
        ion_mode: IonModes,
        logger: Logger = None,
    ):
        """Write binned spectra data on the redis database."""
        self._init_client()
        if logger:
            logger.debug(
                f"Saving {len(binned_spectra)} binned spectra to the client {self.client}"
                f" on hash '{BINNED_SPECTRUM_HASHES}_{ion_mode}'."
            )
        pipe = self.client.pipeline()
        for spectrum in binned_spectra:
            pipe.hset(
                f"{BINNED_SPECTRUM_HASHES}_{ion_mode}",
                spectrum.metadata["spectrum_id"],
                pickle.dumps(spectrum),
            )
        pipe.execute()

    def read_binned_spectra(
        self,
        ion_mode: IonModes,
        spectrum_ids: List[str] = None,
    ) -> List[BinnedSpectrum]:
        """Read the binned spectra from spectra IDs."""
        self._init_client()
        return self._read_hashes(f"{BINNED_SPECTRUM_HASHES}_{ion_mode}", spectrum_ids)
