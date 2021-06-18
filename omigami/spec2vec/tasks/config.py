import datetime

DEFAULT_CONFIG = dict(max_retries=3, retry_delay=datetime.timedelta(seconds=10))


def merge_configs(kwargs):
    return {k: v for k, v in {**DEFAULT_CONFIG.copy(), **kwargs}.items()}
