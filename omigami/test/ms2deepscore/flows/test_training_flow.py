import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from drfs.filesystems import get_fs

from omigami.ms2deepscore.config import SOURCE_URI_PARTIAL_GNPS
from omigami.gateways.data_gateway import SpectrumDataGateway

from omigami.flow_config import (
    make_flow_config,
    PrefectStorageMethods,
    PrefectExecutorMethods,
)
from omigami.ms2deepscore.flows.training_flow import (
    build_training_flow,
    TrainingFlowParameters,
)

from omigami.test.conftest import ASSETS_DIR

os.chdir(Path(__file__).parents[4])

@pytest.fixture
def flow_config():
    flow_config = make_flow_config(
        image="image-ref-name-test-harry-potter-XXII",
        storage_type=PrefectStorageMethods.S3,
        executor_type=PrefectExecutorMethods.LOCAL_DASK,
        redis_db="0",
    )
    return flow_config

#TODO: Needs to be implemented as soon as the pipeline is done
def test_training_flow(flow_config):
    pass

#TODO: Needs to be implemented as soon as the pipeline is done
def test_run_training_flow(
    tmpdir, flow_config, mock_default_config, clean_chunk_files, redis_full_setup
):
    pass