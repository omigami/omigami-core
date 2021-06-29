import confuse

config = confuse.Configuration("omigami/ms2deepscore", __name__)

MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
