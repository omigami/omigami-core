import click

from omigami.spec2vec.cli import spec2vec_cli


@click.group(name="omigami")
def cli():
    pass


cli.add_command(spec2vec_cli)


if __name__ == "__main__":
    cli()
