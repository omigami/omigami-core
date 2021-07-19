from dataclasses import dataclass

from prefect import Task

from omigami.gateways import RedisSpectrumDataGateway, InputDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class SaveRawSpectraParameters:
    """
    Parameters to determine aspects of the SaveRawSpectra task

    spectrum_dgw: RedisSpectrumDataGateway
        A gateway that grands access to the redis database
    input_dgw: InputDataGateway
        A InputDataaGateway that is able to load the gnps dataset
    overwrite_all: bool = False
        If true overwrites all ids in the database that passed to the function. Otherwise only adds new ones.
    """

    spectrum_dgw: RedisSpectrumDataGateway
    overwrite_all: bool = False


class SaveRawSpectra(Task):
    """
    Prefect task to save the raw spectra passed to it.

    save_parameters: SaveRawSpectraParameters
        Parameters determining how certain aspects of the task act.

    Retruns
        Returns all ids
    """

    def __init__(
        self,
        save_parameters: SaveRawSpectraParameters,
        **kwargs,
    ):
        self._spectrum_dgw = save_parameters.spectrum_dgw
        self._overwrite_all = save_parameters.overwrite_all
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    # Note: need to take the spectra ids provided by the prior task
    def run(self, spectra_id_chunk: str = None):
        self.logger.info(f"Chunck of spectra is {len(spectra_id_chunk)}")
        redis_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()

        spectrum_ids = [sp["SpectrumID"] for sp in spectra_id_chunk]

        if self._overwrite_all:
            spectrum_ids_to_add = spectrum_ids
        else:
            spectrum_ids_to_add = set(spectrum_ids) - set(redis_spectrum_ids)

        self.logger.info(f"Need to add new Ids: {len(spectrum_ids_to_add) > 0}")
        if len(spectrum_ids_to_add) > 0:
            self.logger.info(
                f"Adding {len(spectrum_ids_to_add)} spectra to the db; \n"
                f"Overwrite: {self._overwrite_all}"
            )

            db_entries = [
                sp for sp in spectra if sp["SpectrumID"] in spectrum_ids_to_add
            ]
            self._spectrum_dgw.write_raw_spectra(db_entries)

        return spectrum_ids_to_add
