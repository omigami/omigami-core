""" Dress dependency management
This script specifies how we manage dependencies. Conda doesn't allow us to
freeze dependencies cross plattform. The suggestion is to only specify dependencies
that your package relies on directly (just one level) and fix the versions if needed.
This is exactly what dress.py script does. After setting up the environment with
minimal version fixes (as specified in environment.yaml) you can freeze these with
python dress.py env freeze environment.yaml
This will run a conda env export command and extract all the versions that were
installed. Then it will relax the versions, by replacing the patch version with '*'.
Finally it will filter to only packages mentioned in the original environment.yaml
file and generate a new environment.frozen.yaml.
This way we should always get the latest bugfixes, but not risk to include any
breaking changes. Updating all packages should also be straightforward by recreating
the environment as described above. Finally we don't rely on pip anymore at all
(except for installing the package itself).
"""
import os
import re
import subprocess

import click
import yaml

try:
    import conda.cli.python_api as conda
except:
    conda = object()


class CondaResult:
    def __init__(self, stdout: str, stderr: str, return_code: int):
        self.stdout = stdout
        self.stderr = stderr
        self.return_code = return_code


@click.group()
def cli():
    pass


@cli.group()
def env():
    pass


@env.command()
@click.argument("requirements")
def optimize(requirements):
    """Optimize a requirements.txt by moving packages to conda."""
    missing = []
    available = []
    with open(requirements, "r") as fp:
        for package in list(fp.readlines()):
            package = package.strip()
            res = _run_conda(
                conda.Commands.SEARCH,
                "-c",
                "conda-forge",
                "-c",
                "bioconda",
                package,
                use_exception_handler=True,
            )
            if res.return_code != 0:
                missing.append(package)
            else:
                available.append(package)

    click.echo("The following packages are not on conda")
    click.echo("\n".join(missing))

    click.echo("Add these to conda:")
    click.echo(yaml.dump(available))


def _run_conda(*args, **kwargs) -> CondaResult:
    (stdout_str, stderr_str, return_code_int) = conda.run_command(*args, **kwargs)
    return CondaResult(stdout_str, stderr_str, return_code_int)


@env.command()
@click.argument("environment_file")
@click.option("--freeze-pip/--no-freeze-pip", default=False)
def freeze(environment_file, freeze_pip):
    """Provide cross plattform freeze for conda environments."""
    dir_ = os.path.dirname(os.path.abspath(environment_file))
    with open(environment_file, "rb") as fp:
        env_spec = yaml.safe_load(fp)
    res = subprocess.run(["conda", "config", "--show"], stdout=subprocess.PIPE)
    conda_config = yaml.load(res.stdout)

    res = subprocess.run(
        ["conda", "env", "export", "-n", env_spec["name"]], stdout=subprocess.PIPE
    )
    plattform_versions = yaml.safe_load(res.stdout)
    versions = {}
    for dep in plattform_versions["dependencies"]:
        if not isinstance(dep, str):
            continue
        pckg, version = dep.split("=")[:-1]
        version = version.split(".")
        version[-1] = "*"
        version = ".".join(version)
        versions[pckg.lower()] = version

    frozen = []
    for dep in env_spec["dependencies"]:
        if not isinstance(dep, str):
            frozen.append(dep)
            continue
        split_res = re.split(r"[><=]", dep)
        if len(split_res) == 2:
            pckg, version = split_res
        elif len(split_res) == 3:
            pckg, version = split_res[:-1]
        elif len(split_res) == 1:
            pckg = dep
            version = ""
        else:
            raise ValueError(f"Unexpected requirement found: {dep}")
        if version:
            frozen.append(dep)
            continue

        try:
            version = versions[pckg.lower()]
        except KeyError:
            click.echo(f"Could not find version for {pckg}. Please adjust manually")
        frozen.append(f"{pckg}={version}")

    env_spec["dependencies"] = frozen
    conda_out_path = os.path.join(dir_, "environment.frozen.yaml")
    with open(conda_out_path, "w") as fp:
        yaml.dump(env_spec, fp, default_flow_style=False, indent=4)
    click.echo(f"Written frozen env to {conda_out_path}")

    if freeze_pip:
        pip_out_path = os.path.join(dir_, "requirements.frozen.txt")
        pip_executable = os.path.join(
            conda_config["envs_dirs"][0], env_spec["name"], "bin", "pip"
        )
        res = subprocess.run(
            [pip_executable, "list", "--format", "freeze"], stdout=subprocess.PIPE
        )
        with open(pip_out_path, "wb") as fp:
            fp.write(res.stdout)
        click.echo(f"Written frozen requirements to {pip_out_path}")


if __name__ == "__main__":
    cli()
