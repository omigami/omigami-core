import click

from omigami.ms2deepscore.cli import ms2deepscore_cli
from omigami.spectra_matching.spec2vec.cli import spec2vec_cli


@click.group(name="omigami")
def cli():
    pass


cli.add_command(spec2vec_cli)
cli.add_command(ms2deepscore_cli)


if __name__ == "__main__":
    cli()
