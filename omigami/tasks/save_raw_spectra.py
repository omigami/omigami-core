from dataclasses import dataclass
from typing import List
from prefect import Task
from omigami.gateways import RedisSpectrumDataGateway, InputDataGateway
from omigami.spectrum_cleaner import SpectrumCleaner
from omigami.utils import merge_prefect_task_configs


@dataclass
class SaveRawSpectraParameters:
    """
    Parameters to determine aspects of the SaveRawSpectra task

    spectrum_dgw: RedisSpectrumDataGateway
        A gateway that grands access to the redis database
    input_dgw: InputDataGateway
        A InputDataaGateway that is able to load the gnps dataset
    cleaner: SpectrumCleaner
        A SpectrumCleaner that performs some basic cleaning
    overwrite_all: bool = False
        If true overwrites all ids in the database that passed to the function. Otherwise only adds new ones.
    """

    spectrum_dgw: RedisSpectrumDataGateway
    input_dgw: InputDataGateway
    cleaner: SpectrumCleaner
    overwrite_all: bool = False


class SaveRawSpectra(Task):
    """
    Prefect task to save the raw spectra passed to it.

    Parameters:
    -----------
    save_parameters: SaveRawSpectraParameters
        Parameters determining how certain aspects of the task act.
    """

    def __init__(
        self,
        save_parameters: SaveRawSpectraParameters,
        **kwargs,
    ):
        self._spectrum_dgw = save_parameters.spectrum_dgw
        self.input_dgw = save_parameters.input_dgw
        self._cleaner = save_parameters.cleaner
        self._overwrite_all = save_parameters.overwrite_all
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    def _get_spectrum_from_list(self, spec_list, id):

        try:
            for spec in spec_list:
                if spec["spectrum_id"] == id:
                    return spec
        except:
            print("except")
            for spec in spec_list:
                if spec.metadata["spectrum_id"] == id:
                    return spec

    def run(self, gnps_path: str = None) -> List[str]:
        """The run method of a Prefect task takes a path to arw gnps json data and
        saves it as object of the class matchms.Spectrum to a database.

        Parameters:
        ----------
        gnps_path: str
            A string leading to a json datafile containing spectrum data

        Returns:
            A list of all the spectrum_ids contained in the files data
        """
        self.logger.info(f"Loading spectra from {gnps_path}")
        spectra_from_file = self.input_dgw.load_spectrum(gnps_path)

        redis_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()
        spectrum_ids = [sp["spectrum_id"] for sp in spectra_from_file]

        if self._overwrite_all:
            spectrum_ids_to_add = spectrum_ids
        else:
            spectrum_ids_to_add = set(spectrum_ids) - set(redis_spectrum_ids)

        self.logger.info(f"Need to add new Ids: {len(spectrum_ids_to_add) > 0}")
        if len(spectrum_ids_to_add) > 0:
            self.logger.info(
                f"Adding {len(spectrum_ids_to_add)} spectra to the db \n"
                f"Overwrite: {self._overwrite_all}"
            )

            db_entries = [
                sp
                for sp in spectra_from_file
                if sp["spectrum_id"] in spectrum_ids_to_add
            ]

            spectra_to_save = self._cleaner.clean(db_entries)

            self._spectrum_dgw.write_raw_spectra(spectra_to_save)

        return spectrum_ids
