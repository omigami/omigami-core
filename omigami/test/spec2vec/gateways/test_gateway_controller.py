import pytest
import os

from omigami.gateways.fs_data_gateway import FSDataGateway
from omigami.ms2deepscore.config import PROJECT_NAME
from omigami.spec2vec.gateways.gateway_controller import Spec2VecGatewayController
from omigami.spec2vec.gateways.redis_spectrum_gateway import (
    Spec2VecRedisSpectrumDataGateway,
)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_write_documents(s3_documents_directory, documents_data, s3_mock):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    fs_dgw = FSDataGateway()

    document_dgw_controller = Spec2VecGatewayController(
        redis_dgw,
        fs_dgw,
        ion_mode=ion_mode,
    )

    if not fs_dgw.exists(s3_documents_directory):
        fs_dgw.makedirs(s3_documents_directory)

    document_dgw_controller.write_documents(
        f"{s3_documents_directory}/test.pickle",
        documents_data,
    )

    assert len(fs_dgw.listdir(s3_documents_directory)) == 1
    assert len(redis_dgw.list_document_ids(ion_mode)) == len(documents_data)


@pytest.mark.skipif(
    os.getenv("SKIP_REDIS_TEST", True),
    reason="It can only be run if the Redis is up",
)
def test_remove_documents(s3_documents_directory, documents_stored, s3_mock):
    ion_mode = "positive"
    redis_dgw = Spec2VecRedisSpectrumDataGateway(PROJECT_NAME)
    fs_dgw = FSDataGateway()

    document_dgw_controller = Spec2VecGatewayController(
        redis_dgw,
        fs_dgw,
        ion_mode=ion_mode,
    )
    for doc_path in fs_dgw.listdir(s3_documents_directory):
        document_dgw_controller.remove_documents_file(doc_path)

    assert len(fs_dgw.listdir(s3_documents_directory)) == 0
    assert len(redis_dgw.list_document_ids(ion_mode)) == 0
