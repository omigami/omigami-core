import confuse


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/ms2deepscore"


config = Configuration("omigami/ms2deepscore", __name__)

MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
