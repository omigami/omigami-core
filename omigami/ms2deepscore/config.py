import confuse

config = confuse.Configuration("omigami/ms2deepscore", __name__)


MS2DEEPSCORE_MODEL_URI = config["model_uri"].get(str)
MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
