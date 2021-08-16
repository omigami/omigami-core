import os
import tempfile
from typing import Optional

import h5py
from drfs import DRPath
from drfs.filesystems import get_fs
from drfs.filesystems.base import FileSystemBase
from drfs.filesystems.local import LocalFileSystem
from drfs.filesystems.s3 import S3FileSystem
from ms2deepscore import SpectrumBinner
from ms2deepscore.models import SiameseModel
from tensorflow import keras
from tensorflow.python.keras.saving import hdf5_format

from omigami.gateways.fs_data_gateway import FSDataGateway


class MS2DeepScoreFSDataGateway(FSDataGateway):
    def __init__(self, fs: Optional[FileSystemBase] = None):
        super().__init__(fs)

    def save(self, model: SiameseModel, output_path: str):
        path = DRPath(output_path)
        if self.fs is None:
            self.fs = get_fs(path)

        if isinstance(self.fs, LocalFileSystem):
            model.save(path)
        elif isinstance(self.fs, S3FileSystem):
            _, tmp_path = tempfile.mkstemp()
            model.save(tmp_path)
            try:
                self.put(tmp_path, path)
            finally:
                os.remove(tmp_path)

    def load_model(
        self,
        model_path: str,
    ) -> SiameseModel:
        path = DRPath(model_path)
        if self.fs is None:
            self.fs = get_fs(path)

        with h5py.File(self.fs.open(path, "rb"), mode="r") as f:
            binner_json = f.attrs["spectrum_binner"]
            keras_model = hdf5_format.load_model_from_hdf5(f)

        spectrum_binner = SpectrumBinner.from_json(binner_json)

        assert isinstance(
            keras_model, keras.Model
        ), f"Expected keras model as input, got {type(keras_model)}"

        return SiameseModel(spectrum_binner, keras_model=keras_model)
