import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List

from prefect import Task

from omigami.spectra_matching.storage import DataGateway
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


@dataclass
class DownloadParameters:
    """
    Dataclass to specify parameters to download a dataset with the Download Task

    source_uri: str
        URI to the dataset
    output_directory: str
        Output directory for the downloaded data
    file_name: str = "gnps.json"
        Filename of the dataset
    checkpoint_file: str = "spectrum_ids.pkl"
        Filename of the checkpointfile
    """

    source_uri: str
    output_directory: str
    file_name: str = "gnps.json"
    checkpoint_file: str = "spectrum_ids.pkl"

    @property
    def download_path(self):
        return f"{self.output_directory}/{self.file_name}"

    @property
    def checkpoint_path(self):
        return f"{self.output_directory}/{self.checkpoint_file}"

    @property
    def kwargs(self):
        return dict(
            input_uri=self.source_uri,
            download_path=self.download_path,
            checkpoint_path=self.checkpoint_path,
        )


class DownloadData(Task):
    def __init__(
        self,
        data_gtw: DataGateway,
        download_parameters: DownloadParameters,
        **kwargs,
    ):
        self._data_gtw = data_gtw
        self.input_uri = download_parameters.source_uri
        self.download_path = download_parameters.download_path
        self.checkpoint_path = download_parameters.checkpoint_path

        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
            **create_prefect_result_from_path(download_parameters.checkpoint_path),
        )

    def refresh_download(
        self, download_path: str, time_threshold_days: int = 30
    ) -> bool:
        """
        Checks if the data has already been downloaded and is older then time_threshold_days.

        Parameters:
        ----------
        download_path: str
            Path to the downloaded file
        time_threshold_days: int
            Threshold that determines when the data should be downloaded again.

        Returns: True or False depending of the files present and the time threshold.

        """
        downloaded_files = pathlib.Path(download_path)

        if not downloaded_files.exists():
            return True

        file_time = datetime.fromtimestamp(downloaded_files.stat().st_mtime)

        if (datetime.now() - file_time).days >= time_threshold_days:
            return True

        return False

    def run(self) -> List[str]:
        """
        Prefect task to download and save GNPS data into filesystem, only if data is
        NOT up-to-date with the current GNPS data.

        Returns
        -------
        spectrum ids: List[str]
            List of spectrum_ids

        """

        if self.refresh_download(self.download_path):
            self._data_gtw.download_gnps(self.input_uri, self.download_path)
        spectrum_ids = self._data_gtw.get_spectrum_ids(self.download_path)
        self.logger.info(
            f"Downloaded {len(spectrum_ids)} spectra from {self.input_uri} to {self.download_path}."
        )

        self.logger.info(f"Saving spectrum ids to {self.checkpoint_path}")
        self._data_gtw.serialize_to_file(self.checkpoint_path, spectrum_ids)

        return spectrum_ids
