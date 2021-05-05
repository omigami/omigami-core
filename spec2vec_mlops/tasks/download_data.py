from prefect import Task

from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(self, input_dgw: InputDataGateway):
        self._input_dgw = input_dgw
        super().__init__(**DEFAULT_CONFIG)

    # TODO: refactor to use prefect's checkpoint functionality
    def run(
        self, input_uri: str = None, output_dir: str = None, dataset_id: str = None
    ) -> str:
        file_path = self._input_dgw.download_gnps(input_uri, output_dir, dataset_id)
        return file_path
