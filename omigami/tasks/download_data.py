import pathlib
from dataclasses import dataclass
from datetime import datetime
from typing import List

from prefect import Task

from omigami.gateways.data_gateway import DataGateway
from omigami.utils import create_prefect_result_from_path, merge_prefect_task_configs


@dataclass
class DownloadParameters:
    """
    Dataclass to specify parameters to download a dataset with the Download Task

    Parameters
    ----------
    input_uri: str
        URI to the dataset
    output_dir: str
        Output directory for the downloaded data
    dataset_id: str
        ID of the dataset
    dataset_file: str = "gnps.json"
        Filename of the dataset
    checkpoint_file: str = "spectrum_ids.pkl"
        Filename of the checkpointfile
    -------
    """

    input_uri: str
    output_dir: str
    dataset_id: str
    dataset_file: str = "gnps.json"
    checkpoint_file: str = "spectrum_ids.pkl"

    def __post_init__(self):
        self.directory = f"{self.output_dir}/{self.dataset_id}"

    @property
    def download_path(self):
        return f"{self.directory}/{self.dataset_file}"

    @property
    def checkpoint_path(self):
        return f"{self.directory}/{self.checkpoint_file}"

    @property
    def kwargs(self):
        return dict(
            input_uri=self.input_uri,
            download_path=self.download_path,
            checkpoint_path=self.checkpoint_path,
        )


class DownloadData(Task):
    """
    Prefect Task for downloading and saving a Dataset.

    Parameters:
    ----------
    data_gtw: InputDataGateway
        Class that holds the functions and requirments to download the dataset in need
    download_parameters: DownloadParameters
        Dataclass holding variables for the downloading process
    ----------

    """

    def __init__(
        self,
        data_gtw: DataGateway,
        download_parameters: DownloadParameters,
        **kwargs,
    ):
        self._data_gtw = data_gtw
        self.input_uri = download_parameters.input_uri
        self.download_path = download_parameters.download_path
        self.checkpoint_path = download_parameters.checkpoint_path

        config = merge_prefect_task_configs(kwargs)

        super().__init__(
            **config,
            **create_prefect_result_from_path(download_parameters.checkpoint_path),
            checkpoint=True,
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
        This task downloads GNPS data into S3 filesystem, if only data is not up-to-date
        with the current GNPS data.

        Returns
        -------
        spectrum ids: List of spectrum_ids saved to filesystem

        """

        if self.refresh_download(self.download_path):
            self._data_gtw.download_gnps(self.input_uri, self.download_path)

        spectrum_ids = self._data_gtw.get_spectrum_ids(self.download_path)
        self._data_gtw.serialize_to_file(self.checkpoint_path, spectrum_ids)

        self.logger.info(
            f"Downloaded {len(spectrum_ids)} spectra from {self.input_uri} to {self.download_path}."
        )
        self.logger.info(f"Saving spectrum ids to {self.checkpoint_path}")

        return spectrum_ids
