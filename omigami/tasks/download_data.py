from dataclasses import dataclass
from typing import List

from prefect import Task

from omigami.data_gateway import InputDataGateway
from omigami.flows.utils import create_result
from omigami.tasks.config import merge_configs


@dataclass
class DownloadParameters:
    input_uri: str
    output_dir: str
    dataset_id: str
    # these are changed from defaults during tests only
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
    def __init__(
        self,
        input_dgw: InputDataGateway,
        download_parameters: DownloadParameters,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self.input_uri = download_parameters.input_uri
        self.download_path = download_parameters.download_path
        self.checkpoint_path = download_parameters.checkpoint_path

        config = merge_configs(kwargs)

        super().__init__(
            **config,
            **create_result(download_parameters.checkpoint_path),
            checkpoint=True,
        )

    def run(self) -> List[str]:
        self._input_dgw.download_gnps(self.input_uri, self.download_path)
        spectrum_ids = self._input_dgw.get_spectrum_ids(self.download_path)
        self._input_dgw.serialize_to_file(self.checkpoint_path, spectrum_ids)
        self.logger.info(
            f"Downloaded {len(spectrum_ids)} spectra from {self.download_path}."
        )
        self.logger.info(f"Saving spectrum ids to {self.checkpoint_path}")
        return spectrum_ids
