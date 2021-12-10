import click

from omigami.config import SOURCE_URI_PARTIAL_GNPS, STORAGE_ROOT

common_flow_options = [
    click.option(
        "--image", "-i", type=str, required=True, help="Image used to build the flow"
    ),
    click.option(
        "--project-name",
        "-p",
        help="Name of the project. This is used as identification by Prefect",
        required=True,
    ),
    click.option(
        "--flow-name",
        help="Name of the flow. This is used as identification by Prefect",
        required=True,
    ),
]

common_training_options = [
    click.option(  # TODO: these two parameters need some refactoring
        "--dataset-id",
        type=click.Choice(["small", "10k", "complete"]),
        required=True,
        help="Name of the dataset of choice. It must match the source-uri of the dataset.",
    ),
    click.option(
        "--source-uri",
        default=SOURCE_URI_PARTIAL_GNPS,
        help="URI to download training data from. Only downloads if it is not already downloaded",
    ),
    click.option(
        "--ion-mode",
        type=click.Choice(["positive", "negative"]),
        default="positive",
        help="Which ion mode to use.",
        show_default=True,
    ),
    click.option(
        "--deploy-model",
        is_flag=True,
        type=bool,
        default=False,
        help="Flag to whether deploy the model as a seldon deployment or not",
        show_default=True,
    ),
    click.option(
        "--overwrite-model",
        is_flag=True,
        help="Flag to overwrite existing deployed model. Only works with --deploy-model flag",
        show_default=True,
        default=False,
    ),
    click.option(
        "--dataset-directory",
        type=str,
        default=str(STORAGE_ROOT / "datasets"),
        show_default=True,
        help="Directory where the dataset will be downloaded to or where it is, in case "
        "it is already downloaded.",
    ),
]
