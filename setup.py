# coding=utf-8
from setuptools import setup, find_packages

import versioneer

packages = find_packages()

setup(
    name="omigami-core",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description="Omigami core package built using OpenMLOps tools.",
    author="Data Revenue GmbH",
    author_email="markus@datarevenue.com",
    install_requires=[],
    packages=packages,
    package_data={},
    zip_safe=False,
    entry_points="""
        [console_scripts]
        omigami=omigami.cli:cli
    """,
)

print(
    """
    WARNING
    -------
    
    Will not install any dependencies. Please manage dependencies using conda.
    """
)
