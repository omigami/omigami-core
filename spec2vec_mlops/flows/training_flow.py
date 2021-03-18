import click
from prefect import Flow, Parameter, Client
from prefect.engine.state import State
from prefect.run_configs import KubernetesRun
from prefect.storage import S3

from spec2vec_mlops import config
from spec2vec_mlops.tasks.load_data import load_data_task

# variable definitions region
SOURCE_URI_COMPLETE_GNPS = config["gnps_json"]["uri_complete_data"].get(str)
SOURCE_URI_PARTIAL_GNPS = config["gnps_json"]["uri_partial_data"].get(str)
API_SERVER = "https://prefect.mlops.datarevenue.com/graphql"  # "http://localhost:4200"
# setting API_SERVER to "http://localhost:4200" requires port-forwarding


def spec2vec_train_pipeline_local(source_uri: str) -> State:
    with Flow("flow") as flow:
        raw = load_data_task(source_uri)
    state = flow.run()
    return state


def spec2vec_train_pipeline_distributed(
    source_uri: str = SOURCE_URI_COMPLETE_GNPS,
    api_server: str = API_SERVER,
    project_name: str = "spec2vec-mlops-project",
) -> str:
    custom_confs = {
        "run_config": KubernetesRun(
            image="drtools:spec2vec-mlops-v1",
            labels=["dev"],
            service_account_name="dask-jupyter-sa",
        ),
        "storage": S3("dr-prefect"),
    }
    with Flow("spec2vec-training-flow", **custom_confs) as training_flow:
        uri = Parameter(name="uri")
        raw = load_data_task(uri)
        print("Data loading is complete.")
        # cleaned = clean_data_task(raw)
        # saved = save_data_to_feast_task(cleaned)
        # documents = convert_data_to_documents_task(saved)
        # encoded = encode_training_data_task(documents)
        # trained = train_model_task(documents)
    client = Client(api_server=api_server)
    client.create_project(project_name)
    training_flow_id = client.register(
        training_flow,
        project_name=project_name,
    )
    flow_run_id = client.create_flow_run(
        flow_id=training_flow_id,
        run_name=f"run {project_name}",
        parameters={"uri": source_uri},
    )
    return flow_run_id


@click.group()
def cli():
    pass


@cli.command(name="register-train-pipeline")
def register_train_pipeline_cli(*args, **kwargs):
    spec2vec_train_pipeline_distributed(*args, **kwargs)


@cli.command(name="register-all-flows")
def deploy_model_cli():
    spec2vec_train_pipeline_distributed()


if __name__ == "__main__":
    cli()
