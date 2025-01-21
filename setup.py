import os
from setuptools import find_packages, setup


def read_requirements(filename: os.PathLike):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


required_deps = read_requirements("requirements.txt")

setup(
    name="plmfit",
    version="0.0.1",
    description="PLMFit",
    long_description="PLMFit",
    author="Thomas Bikias, Evangelos Stamkopoulos",
    packages=find_packages(),
    license="Apache License 2.0",
    zip_safe=True,
    install_requires=required_deps,
    entry_points={
        'console_scripts': [
            'plmfit=plmfit.__main__:main'
        ]
    }
)
