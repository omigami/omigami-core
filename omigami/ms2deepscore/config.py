import confuse

config = confuse.Configuration("omigami/ms2deepscore", __name__)


MS2DEEP_MODEL_URI = config["model_uri"].get(str)
