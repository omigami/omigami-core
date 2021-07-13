from dataclasses import dataclass

from prefect import Task

from omigami.gateways.data_gateway import SpectrumDataGateway
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


@dataclass
class SaveRawSpectraParameters:
    spectrum_dgw: SpectrumDataGateway
    skip_task: bool = False


class SaveRawSpectra(Task):
    def __init__(
        self,
        save_parameters: SaveRawSpectraParameters,
        **kwargs,
    ):

        self._spectrum_dgw = save_parameters.spectrum_dgw
        self._skip = save_parameters.skip_task
        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
        )

    def run(self, gnps_path: str = None):
        # Gets spectrum ids from the redis database
        self._spectrum_dgw.list_spectrum_ids()

        # Loads spectra from the gnps file
        if not self._skip:
            self.logger.info(f"Loading spectra from {gnps_path}")
            spectra = self._input_dgw.load_spectrum(gnps_path)
            spectrum_ids = [sp["SpectrumID"] for sp in spectra]

        # Compare IDs from redis and the file

        # Add only new IDs to the redis DB

        # Return spectrum IDs
        return False
