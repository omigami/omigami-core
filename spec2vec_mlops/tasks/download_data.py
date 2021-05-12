from dataclasses import dataclass
from typing import List

from prefect import Task

from spec2vec_mlops.tasks.config import merge_configs
from spec2vec_mlops.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(
        self,
        input_dgw: InputDataGateway,
        input_uri: str,
        download_path: str,
        checkpoint_path: str,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self.input_uri = input_uri
        self.download_path = download_path
        self.checkpoint_path = checkpoint_path

        config = merge_configs(kwargs)

        super().__init__(
            **config,
            checkpoint=True,
        )

    def run(self) -> List[str]:
        self._input_dgw.download_gnps(self.input_uri, self.download_path)
        spectrum_ids = self._input_dgw.get_spectrum_ids(self.download_path)
        self._input_dgw.save_spectrum_ids(self.checkpoint_path, spectrum_ids)
        return spectrum_ids


@dataclass
class DownloadParameters:
    input_uri: str
    output_dir: str
    dataset_name: str
    input_dgw: InputDataGateway
    checkpoint: str = "spectrum_ids.pkl"

    @property
    def download_path(self):
        return f"{self.output_dir}/{self.dataset_name}"

    @property
    def checkpoint_path(self):
        return f"{self.output_dir}/{self.checkpoint}"

    @property
    def kwargs(self):
        return dict(
            input_uri=self.input_uri,
            download_path=self.download_path,
            input_dgw=self.input_dgw,
            checkpoint_path=self.checkpoint_path,
        )
