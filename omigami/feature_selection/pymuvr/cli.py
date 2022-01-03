"""
PyMUVR feature selection CLI to build and deploy flows.
"""
import click

from omigami.feature_selection.cli import feature_selection_cli


@feature_selection_cli.command(name="pymuvr")
@click.option(
    "--input-dataset-path",
    type=str,
    required=True,
    help="Path to the dataset that will be used on the algorithm",
)
@click.option(
    "--run-id",
    type=str,
    required=False,
    help="Run ID used to save outputs. If not passed, will use prefect flow run ID.",
)
def deploy_pymuvr_flow():
    pass
