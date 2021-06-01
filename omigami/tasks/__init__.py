from omigami.tasks.download_data import DownloadData, DownloadParameters
from omigami.tasks.check_condition import check_condition
from omigami.tasks.process_spectrum import ProcessSpectrum, ProcessSpectrumParameters
from omigami.tasks.seldon import DeployModel
from omigami.tasks.make_embeddings import MakeEmbeddings
from omigami.tasks.register_model import TrainModel, TrainModelParameters
from omigami.tasks.train_model import train_model_task
from omigami.tasks.create_chunks import CreateChunks
