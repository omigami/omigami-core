from spec2vec_mlops.helper_classes.data_loader import DataLoader
from spec2vec_mlops.helper_classes.spec2vec_model import Model


def test_model(gnps_small_json, word2vec_model):
    dl = DataLoader()
    loaded_data = dl.load_gnps_json(gnps_small_json)
    model = Model(word2vec_model, 1)
    model.predict(None, loaded_data)
