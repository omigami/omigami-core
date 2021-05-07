from dataclasses import dataclass
from typing import List

from prefect import Task
from prefect.engine.result import Result

from spec2vec_mlops.tasks.config import merge_configs
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(
        self,
        input_dgw: InputDataGateway,
        input_uri: str,
        download_path: str,
        dataset_name: str,
        result: Result,
        **kwargs,
    ):
        self._input_dgw = input_dgw
        self.input_uri = input_uri
        self.download_path = download_path

        config = merge_configs(kwargs)

        super().__init__(
            **config,
            result=result,
            checkpoint=True,
            target=dataset_name,
        )

    def run(self) -> List[str]:
        self._input_dgw.download_gnps(self.input_uri, self.download_path)
        spectrum_ids = self._input_dgw.get_spectrum_ids(self.download_path)
        return spectrum_ids


@dataclass
class DownloadParameters:
    input_uri: str
    output_dir: str
    dataset_name: str
    input_dgw: InputDataGateway

    @property
    def download_path(self):
        return f"{self.output_dir}/{self.dataset_name}"

    @property
    def kwargs(self):
        return dict(
            input_uri=self.input_uri,
            download_path=self.download_path,
            input_dgw=self.input_dgw,
            dataset_name=self.dataset_name,
        )
