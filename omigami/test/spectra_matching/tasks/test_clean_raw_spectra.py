from unittest.mock import Mock

import numpy as np
import pytest
from matchms import Spectrum
from matchms.importing.load_from_json import as_spectrum
from prefect import Flow

from omigami.spectra_matching.storage import FSDataGateway
from omigami.spectra_matching.tasks import CreateChunks, ChunkingParameters
from omigami.spectra_matching.tasks.clean_raw_spectra import (
    SpectrumCleaner,
    CleanRawSpectra,
    CleanRawSpectraParameters,
)
from omigami.test.spectra_matching.conftest import ASSETS_DIR


@pytest.fixture
def previous_task(clean_chunk_files, local_gnps_small_json):
    data_gtw = FSDataGateway()
    output_directory = ASSETS_DIR / "raw" / "positive"
    chunking_parameters = ChunkingParameters(
        local_gnps_small_json, str(output_directory), 150000, "positive"
    )
    chunk_task = CreateChunks(
        data_gtw=data_gtw, chunking_parameters=chunking_parameters
    )

    return chunk_task


def test_clean_raw_spectra_run(previous_task):
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)
    t = CleanRawSpectra(fs_dgw, params)
    chunk_paths = previous_task.run()

    res = t.run(chunk_paths[0])

    assert {sp.metadata["spectrum_id"] for sp in fs_dgw.read_from_file(res)} == {
        sp["SpectrumID"] for sp in fs_dgw.load_spectrum(chunk_paths[0])
    }


def test_clean_raw_spectra_flow(previous_task, mock_default_config):
    output_dir = ASSETS_DIR / "cleaned"
    fs_dgw = FSDataGateway()
    params = CleanRawSpectraParameters(output_dir)

    with Flow("test-flow") as flow:
        p = previous_task()
        t = CleanRawSpectra(fs_dgw, params).map(p)

    res = flow.run()

    assert res.is_successful()

    # running again to use cached results, and asserting it works
    t._fs_dgw.load_spectrum = Mock()
    res_cached = flow.run()

    assert res_cached.is_successful()
    t._fs_dgw.load_spectrum.assert_not_called()


@pytest.fixture
def spectrum(loaded_data):
    return as_spectrum(loaded_data[0])


@pytest.fixture
def spectrum_negative_intensity():
    return Spectrum(
        mz=np.sort(np.random.rand(216)), intensities=np.random.uniform(-2, 2, 216)
    )


def test_clean_data(spectrum, spectrum_negative_intensity):

    cleaned_data = SpectrumCleaner()._common_cleaning(spectrum)
    cleaned_data_negative_intensity = SpectrumCleaner()._common_cleaning(
        spectrum_negative_intensity
    )
    assert cleaned_data_negative_intensity is None
    assert isinstance(cleaned_data, Spectrum)
    # Asserts invalid inchi keys are set as "" and not N/A, NA, n/a or None
    assert cleaned_data.get("inchi") not in ["N/A", "NA", "n/a", None]
    assert isinstance(cleaned_data.get("charge"), int)
    assert cleaned_data.get("parent_mass")
    assert cleaned_data.get("spectrum_id")


def test_apply_ms2deepscore_filters_negative_intensity(
    spectrum_negative_intensity, spectrum
):
    spectrum_negative_intensity = SpectrumCleaner()._filter_negative_intensities(
        spectrum_negative_intensity
    )
    spectrum = SpectrumCleaner()._filter_negative_intensities(spectrum)

    assert spectrum_negative_intensity is None
    assert spectrum is not None
