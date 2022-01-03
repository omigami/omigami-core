import click

from omigami.cli_options import common_flow_options
from omigami.utils import add_click_options


@click.group("feature-selection")
@add_click_options(common_flow_options)
def feature_selection_cli():
    pass
