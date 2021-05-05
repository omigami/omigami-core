from prefect import Task
from prefect.engine.results import S3Result

from spec2vec_mlops.tasks.config import DEFAULT_CONFIG
from spec2vec_mlops.tasks.data_gateway import InputDataGateway


class DownloadData(Task):
    def __init__(self, input_dgw: InputDataGateway, output_dir: str, dataset_id: str):
        self._input_dgw = input_dgw
        self._target_path = f"{output_dir}/{dataset_id}"
        super().__init__(**DEFAULT_CONFIG, result=S3Result(dir=self._target_path))

    # TODO: refactor to use prefect's checkpoint functionality
    def run(
        self, input_uri: str = None, output_dir: str = None, dataset_id: str = None
    ) -> str:
        file_path = self._input_dgw.download_gnps(input_uri, output_dir, dataset_id)
        return file_path
