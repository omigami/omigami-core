from typing import List

from matchms import Spectrum
from prefect import Task

from omigami.spectra_matching.storage import RedisSpectrumDataGateway, DataGateway
from omigami.utils import merge_prefect_task_configs


class CacheCleanedSpectra(Task):
    def __init__(
        self,
        spectrum_dgw: RedisSpectrumDataGateway,
        fs_dgw: DataGateway,
        **kwargs,
    ):
        self._spectrum_dgw = spectrum_dgw
        self._fs_dgw = fs_dgw
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    def run(self, spectra_path: str = None) -> List[str]:
        """
        Saves cleaned spectra to a cached database that will be later used by the
        seldon deployment.

        Parameters:
        ----------
        spectra_path: str
            A string leading to a json datafile containing spectrum data

        Returns:
            List of cached spectrum IDs

        """
        self.logger.info(f"Loading spectra from {spectra_path}")
        spectra: List[Spectrum] = self._fs_dgw.read_from_file(path=spectra_path)
        spectrum_ids = [sp.metadata["spectrum_id"] for sp in spectra]
        self.logger.info(
            f"Finished loading file. File contains {len(spectrum_ids)} spectra."
        )

        existing_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
        new_spectrum_ids = set(spectrum_ids) - set(existing_spectrum_ids)

        if len(new_spectrum_ids) == 0:
            self.logger.info("There is no new spectra to save.")
        else:
            self.logger.info(f"Saving {len(new_spectrum_ids)} spectra to the database")
            new_spectra = [
                sp for sp in spectra if sp.metadata["spectrum_id"] in new_spectrum_ids
            ]
            self._spectrum_dgw.write_raw_spectra(new_spectra)
            self.logger.info(f"Added {len(new_spectrum_ids)} new spectra to the db.")

        return spectrum_ids
