from typing import List

from prefect import Task

from spectra_matching.storage import DataGateway
from utils import merge_prefect_task_configs


class ListCleanedSpectraPaths(Task):
    def __init__(
        self,
        cleaned_spectra_directory: str,
        fs_dgw: DataGateway,
        **kwargs,
    ):
        self._fs_gtw = fs_dgw
        self._cleaned_spectra_directory = cleaned_spectra_directory

        config = merge_prefect_task_configs(kwargs)
        super().__init__(**config)

    def run(self) -> List[str]:
        """Lists all paths to cleaned spectra files."""
        self.logger.info(
            f"Reading cleaned spectra paths from directory "
            f"{self._cleaned_spectra_directory}."
        )
        cleaned_spectra_paths = self._fs_gtw.list_files(self._cleaned_spectra_directory)
        self.logger.info(f"Found {len(cleaned_spectra_paths)} on directory.")
        return cleaned_spectra_paths
