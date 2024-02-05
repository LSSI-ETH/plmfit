import os
from setuptools import find_packages, setup


def read_requirements(filename: os.PathLike):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


required_deps = read_requirements("requirements.txt")

setup(
    name="plmfit",
    version="0.0.1",
    description="PLMfit",
    long_description="PLMfit",
    author="Thomas Bikias",
    packages=find_packages(),
    license="Apache License 2.0",
    zip_safe=True,
    install_requires=required_deps,
)
