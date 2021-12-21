from pathlib import Path


class Environments:
    docker = "docker"
    local = "local"
    dev = "dev"
    prod = "prod"


Clusters = {
    Environments.local: "",
    Environments.docker: "",
    Environments.dev: "dev",
    Environments.prod: "prod",
}

PrefectServers = {
    Environments.local: "http://localhost:4200/graphql",
    Environments.docker: "http://localhost:4200/graphql",
    Environments.dev: "https://prefect-dev.omigami.com/graphql",
    Environments.prod: "https://prefect.omigami.datarevenue.com/graphql",
}

StorageRoots = {
    Environments.local: str(Path(__file__).parents[1] / "local-deployment" / "results"),
    Environments.docker: "/opt/omigami/local-deployment",
    Environments.dev: "s3://omigami-dev",
    Environments.prod: "s3://omigami",
}

RedisDatabases = {
    Environments.local: {"small": "0", "small_500": "0"},
    Environments.docker: {"small": "0", "small_500": "0"},
    Environments.dev: {"small": "2", "small_500": "2", "10k": "1", "complete": "0"},
    Environments.prod: {"small": "2", "complete": "0"},
}

RedisHosts = {
    Environments.local: "localhost",
    Environments.docker: "host.docker.internal",
    Environments.dev: "redis-master.redis",
    Environments.prod: "redis-master.redis",
}
