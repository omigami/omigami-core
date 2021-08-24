from typing import List, Optional

from spec2vec import SpectrumDocument

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.config import PROJECT_NAME
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


class Spec2VecGatewayController:
    def __init__(self, ion_mode: str, project=PROJECT_NAME):
        self._redis_gateway = Spec2VecRedisSpectrumDataGateway(project=project)
        self._fs_gateway = FSDataGateway()
        self._ion_mode = ion_mode

    def write_documents(self, path: str, documents: List[Optional[SpectrumDocument]]):

        self._redis_gateway.write_document_ids(documents, self._ion_mode)
        self._fs_gateway.serialize_to_file(path, documents)
