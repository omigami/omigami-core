from ._version import get_versions
import confuse

__version__ = get_versions()["version"]
del get_versions

ENV = confuse.Configuration("spec2vec_mlops")
