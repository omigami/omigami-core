import os

import yaml

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions

with open(
    os.path.join(os.path.dirname(__file__), "config_default.yaml")
) as yaml_config_file:
    default_configs = yaml.safe_load(yaml_config_file)
