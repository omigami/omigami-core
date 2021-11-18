import os

import redis

REDIS_HOST = str(os.getenv("REDIS_HOST"))
REDIS_DB = str(os.getenv("REDIS_DB"))
client = None


def get_redis_client():
    global client
    if client is None:
        client = redis.StrictRedis(host=REDIS_HOST, db=REDIS_DB)
    return client


class RedisDataGateway:
    def __init__(self, project: str = None):
        # We initialize it with None so we can pickle this gateway when deploying the flow

        self.client = None
        self.project = project

    def _init_client(self):
        if self.client is None:
            self.client = get_redis_client()

    def _format_redis_key(self, hashes: str, ion_mode: str, run_id: str = None):

        if not run_id:
            return f"{hashes}_{self.project}_{ion_mode}"

        return f"{hashes}_{self.project}_{ion_mode}_{run_id}"
