from typing import List, Optional

from spec2vec import SpectrumDocument

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


class Spec2VecGatewayController:
    def __init__(
        self,
        redis_dgw: Spec2VecRedisSpectrumDataGateway,
        fs_dgw: FSDataGateway,
        ion_mode: str,
    ):
        self._redis_dgw = redis_dgw
        self._fs_dgw = fs_dgw
        self._ion_mode = ion_mode

    def write_documents(self, path: str, documents: List[Optional[SpectrumDocument]]):

        self._redis_dgw.write_document_ids(documents, self._ion_mode)
        self._fs_dgw.serialize_to_file(path, documents)
