from omigami.config import DEFAULT_CONFIG


def add_options(options):
    def _add_options(func):
        for option in reversed(options):
            func = option(func)
        return func

    return _add_options


def merge_configs(kwargs):
    return {k: v for k, v in {**DEFAULT_CONFIG.copy(), **kwargs}.items()}
