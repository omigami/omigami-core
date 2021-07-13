import pytest

from omigami.ms2deepscore.gateways.redis_spectrum_data_gateway import (
    RedisSpectrumDataGateway,
)
from omigami.tasks.save_raw_spectra import SaveRawSpectra, SaveRawSpectraParameters

from prefect import Flow


def test_save_raw_spectra():
    # Setup Test
    spectrum_dgw = RedisSpectrumDataGateway()

    parameters = SaveRawSpectraParameters()
    parameters.spectrum_dgw = spectrum_dgw

    # Run Functions
    with Flow("test-flow") as test_flow:
        raw_spectra = SaveRawSpectra(save_parameters=parameters)()

    res = test_flow.run()
    data = res.result[raw_spectra].result

    # Test Results
    assert res.is_successful()
