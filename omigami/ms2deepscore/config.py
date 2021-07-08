import confuse


class Configuration(confuse.Configuration):
    def config_dir(self):
        return "omigami/ms2deepscore"


config = Configuration("omigami/ms2deepscore", __name__)
SOURCE_URI_PARTIAL_GNPS = config["gnps_uri"]["partial"].get(str)
MODEL_DIRECTORIES = config["storage"]["model_folder"].get(dict)
