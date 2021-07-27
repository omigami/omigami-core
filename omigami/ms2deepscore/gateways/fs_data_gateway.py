import os
import tempfile
from typing import Optional

from drfs import DRPath
from drfs.filesystems import get_fs
from drfs.filesystems.base import FileSystemBase
from drfs.filesystems.local import LocalFileSystem
from drfs.filesystems.s3 import S3FileSystem
from ms2deepscore.models import SiameseModel
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
