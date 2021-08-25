from dataclasses import dataclass
from typing import List

from prefect import Task

from omigami.gateways import RedisSpectrumDataGateway, DataGateway
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
        """The run method of a Prefect task takes a path to raw gnps json data and
        saves it as object of the class matchms.Spectrum to a database.

        Parameters:
        ----------
        gnps_path: str
            A string leading to a json datafile containing spectrum data

        Returns:
            A list of all the spectrum_ids contained in the files data that
            passed the cleaning process.
        """
        self.logger.info(f"Loading spectra from {gnps_path}")
        spectra_from_file = self._data_gtw.load_spectrum(gnps_path)
        spectrum_ids = [sp["spectrum_id"] for sp in spectra_from_file]

        redis_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()

        if self._overwrite_all_spectra:
            spectrum_ids_to_add = spectrum_ids
        else:
            spectrum_ids_to_add = set(spectrum_ids) - set(redis_spectrum_ids)

        spectrum_ids_already_added = list(set(spectrum_ids) - set(spectrum_ids_to_add))

        self.logger.info(f"Need to add new IDs: {len(spectrum_ids_to_add) > 0}")
        if len(spectrum_ids_to_add) > 0:
            self.logger.info(
                f"Cleaning {len(spectrum_ids_to_add)} spectra before adding to db\n"
                f"Overwrite: {self._overwrite_all_spectra}"
            )

            spectra_to_add = [
                sp
                for sp in spectra_from_file
                if sp["spectrum_id"] in spectrum_ids_to_add
            ]

            spectra_to_add = self._spectrum_cleaner.clean(spectra_to_add)

            self._spectrum_dgw.write_raw_spectra(spectra_to_add)

            cleaned_spectrum_ids_to_add = [
                sp.metadata["spectrum_id"] for sp in spectra_to_add
            ]

            self.logger.info(
                f"Adding {len(cleaned_spectrum_ids_to_add)} spectra to the db"
            )

            return cleaned_spectrum_ids_to_add + spectrum_ids_already_added

        return spectrum_ids
