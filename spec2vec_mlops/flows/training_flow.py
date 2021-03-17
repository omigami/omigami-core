from pathlib import Path

import click
from prefect import Flow, Parameter
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.tasks.load_data import load_data_task

# variable definitions region
TEST_URI_PATH = str(Path(__file__).parents[1] / "test" / "assets" / "SMALL_GNPS.json")
TEST_URI = f"file://{TEST_URI_PATH}"
SOURCE_URI = config["gnps_json"]["uri"].get(str)
URI = "https://raw.githubusercontent.com/MLOps-architecture/share/main/test_data/SMALL_GNPS.json"  # TEST_URI  # SOURCE_URI
# TODO: change it to source URI


def spec2vec_train_pipeline():
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools:spec2vec-mlops-v1",
        ),
        "storage": S3("dr-prefect"),
    }
    with Flow("spec2vec-training-flow", **custom_confs) as training_flow:
        uri = Parameter(URI)
        raw = load_data_task(uri)
        print("Data loading is complete...")
        # cleaned = clean_data_task(raw)
        # saved = save_data_to_feast_task(cleaned)
        # documents = convert_data_to_documents_task(saved)
        # encoded = encode_training_data_task(documents)
        # trained = train_model_task(documents)
    training_flow_id = training_flow.register(project_name="spec2vec-mlops-trial-01")
    return training_flow_id


# training_flow_state = training_flow.run()


@click.group()
def cli():
    pass


@cli.command(name="register-train-pipeline")
def register_train_pipeline_cli(*args, **kwargs):
    spec2vec_train_pipeline(*args, **kwargs)


@cli.command(name="register-all-flows")
def deploy_model_cli():
    spec2vec_train_pipeline()


if __name__ == "__main__":
    cli()
