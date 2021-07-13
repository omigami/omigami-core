import pytest
from omigami.tasks.save_raw_spectra import SaveRawSpectra

from prefact import Flow


def test_download_data(mock_default_config, tmpdir):

    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra()()

    res = test_flow.run()

    assert res.is_successful()
