from spec2vec_mlops.tasks.download_data import DownloadData, DownloadParameters
from spec2vec_mlops.tasks.process_spectrum import (
    ProcessSpectrum,
    ProcessSpectrumParameters,
)
from spec2vec_mlops.tasks.seldon import DeployModel
from spec2vec_mlops.tasks.make_embeddings import MakeEmbeddings
from spec2vec_mlops.tasks.register_model import RegisterModel
from spec2vec_mlops.tasks.create_chunks import CreateChunks
from spec2vec_mlops.tasks.train_model import TrainModel, TrainModelParameters
