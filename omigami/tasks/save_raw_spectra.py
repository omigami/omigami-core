from dataclasses import dataclass
from typing import List

from prefect import Task

from omigami.base.gateways import RedisSpectrumDataGateway, DataGateway
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.utils import merge_prefect_task_configs


@dataclass
class SaveRawSpectraParameters:
    """
    Parameters to determine aspects of the SaveRawSpectra task

    spectrum_dgw: RedisSpectrumDataGateway
        A gateway that grants access to the redis database
    data_gtw: InputDataGateway
        A InputDataaGateway that is able to load the gnps dataset
    spectrum_cleaner: SpectrumCleaner
        A SpectrumCleaner that performs some common cleaning
    overwrite_all_spectra: bool = False
        If True, it overwrites all Spectra stored in the database that are passed to
        the function. Otherwise only adds new ones.
    """

    spectrum_dgw: RedisSpectrumDataGateway
    data_gtw: DataGateway
    spectrum_cleaner: SpectrumCleaner
    overwrite_all_spectra: bool = False


class SaveRawSpectra(Task):
    """
    Prefect task to save the raw spectra passed to it.
    """

    def __init__(
        self,
        save_parameters: SaveRawSpectraParameters,
        **kwargs,
    ):
        self._spectrum_dgw = save_parameters.spectrum_dgw
        self._data_gtw = save_parameters.data_gtw
        self._spectrum_cleaner = save_parameters.spectrum_cleaner
        self._overwrite_all_spectra = save_parameters.overwrite_all_spectra
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    def run(self, gnps_path: str = None) -> List[str]:
        """
        Prefect task to clean and save `Spectrum` (matchms object) to REDIS DB. Data
        from `gnps_path` is read, cleaned and then saved to REDIS DB as raw spectra.

        If `_overwrite_all_spectra` is True, then DB is overwritten with data from
        `gnps_path`.

        If `_overwrite_all_spectra` is False, then only new spectra is saved to DB.

        Parameters:
        ----------
        gnps_path: str
            A string leading to a json datafile containing spectrum data

        Returns:
            A list of all the spectrum_ids contained in the files data that
            passed the cleaning process.

        """
        self.logger.info(f"Loading spectra from {gnps_path}")
        gnps_spectra = self._data_gtw.load_spectrum(gnps_path)
        gnps_spectrum_ids = [sp["spectrum_id"] for sp in gnps_spectra]

        if self._overwrite_all_spectra:
            new_spectrum_ids = gnps_spectrum_ids
            existing_spectrum_ids = []
        else:
            existing_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
            new_spectrum_ids = set(gnps_spectrum_ids) - set(existing_spectrum_ids)

        if len(new_spectrum_ids) == 0:
            self.logger.info("There is no new spectra to save.")
            return gnps_spectrum_ids
        else:
            self.logger.info(
                f"Cleaning {len(new_spectrum_ids)} spectra before adding to db\n"
                f"Overwrite: {self._overwrite_all_spectra}"
            )
            new_spectra = [
                sp for sp in gnps_spectra if sp["spectrum_id"] in new_spectrum_ids
            ]
            clean_spectra = self._spectrum_cleaner.clean(new_spectra)
            self._spectrum_dgw.write_raw_spectra(clean_spectra)
            clean_spectrum_ids = [sp.metadata["spectrum_id"] for sp in clean_spectra]
            self.logger.info(f"Added {len(clean_spectrum_ids)} spectra to the db")
            return clean_spectrum_ids + existing_spectrum_ids
