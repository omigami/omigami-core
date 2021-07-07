import os
import pytest
from ms2deepscore import BinnedSpectrum

from omigami.ms2deepscore.helper_classes.spectrum_binner import SpectrumBinner
from omigami.test.conftest import ASSETS_DIR

pytestmark = pytest.mark.skipif(
    not os.path.exists(
        str(
            ASSETS_DIR
            / "ms2deepscore"
            / "pretrained"
            / "MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5"
        )
    ),
    reason="MS2DeepScore_allGNPSpositive_10k_500_500_200.hdf5 is git ignored. Please "
    "download it from https://zenodo.org/record/4699356#.YNyD-2ZKhcA",
)


def test_bin_spectra(cleaned_data_ms2deep_score, ms2deepscore_real_model_path):
    spectrum_binner = SpectrumBinner()
    binned_spectra = spectrum_binner.bin_spectra(
        cleaned_data_ms2deep_score, model_path=ms2deepscore_real_model_path
    )

    assert isinstance(binned_spectra[0], BinnedSpectrum)
    assert (
        binned_spectra[1].metadata["spectrum_id"]
        == cleaned_data_ms2deep_score[1].metadata["spectrum_id"]
    )
