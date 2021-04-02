import confuse

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

# loads default config_default.yaml file lazily
config = confuse.Configuration("spec2vec_mlops", __name__)
