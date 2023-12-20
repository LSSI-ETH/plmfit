import os
from pathlib import Path
from setuptools import find_packages, setup


def parse_requirements(filename: os.PathLike):
    with open(filename) as f:
        requirements = f.read().splitlines()

        def extract_url(line):
            return next(filter(lambda x: x[0] != "-", line.split()))

        extra_URLs = []
        deps = []
        for line in requirements:
            if line.startswith("#") or line.startswith("-r"):
                continue

            # handle -i and --extra-index-url options
            if "-i " in line or "--extra-index-url" in line:
                extra_URLs.append(extract_url(line))
            else:
                deps.append(line)
    return deps, extra_URLs


required_deps, extra_URLs = parse_requirements(Path("requirements.txt"))

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
