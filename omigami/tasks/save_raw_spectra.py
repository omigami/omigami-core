from dataclasses import dataclass

from prefect import Task

from omigami.gateways.data_gateway import SpectrumDataGateway, InputDataGateway
from omigami.utils import merge_prefect_task_configs


@dataclass
class SaveRawSpectraParameters:
    spectrum_dgw: SpectrumDataGateway
    input_dgw: InputDataGateway
    skip_task: bool = False


class SaveRawSpectra(Task):
    def __init__(
        self,
        save_parameters: SaveRawSpectraParameters,
        **kwargs,
    ):
        self._spectrum_dgw = save_parameters.spectrum_dgw
        self._input_dgw = save_parameters.input_dgw
        self._skip = save_parameters.skip_task
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    def run(self, gnps_path: str = None):
        # Gets spectrum ids from the redis database
        self.logger.info(f"Loading spectra from redis")
        redis_spectrum_ids = self._spectrum_dgw.list_spectrum_ids()

        if self._skip:
            return redis_spectrum_ids

        # Loads spectra from the gnps file
        self.logger.info(f"Loading spectra from {gnps_path}")
        spectra = self._input_dgw.load_spectrum(gnps_path)
        spectrum_ids = [sp["SpectrumID"] for sp in spectra]

        # Compare IDs from redis and the file
        new_spectrum_ids = set(redis_spectrum_ids) - set(spectrum_ids)
        if len(new_spectrum_ids) > 0:
            self.logger.info(
                f"Discrepancy between stored and downloaded data is {len(new_spectrum_ids)} "
            )

            # Add only new IDs to the redis DB
            new_db_entries = [
                sp for sp in spectra if sp["SpectrumID"] in new_spectrum_ids
            ]
            result = self._spectrum_dgw.write_raw_spectra(new_db_entries)

            self.logger.info(f"Adding new ids: {result}")

        # Return spectrum IDs
        return self._spectrum_dgw.list_spectrum_ids()
