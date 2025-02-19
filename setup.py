import os
from setuptools import find_packages, setup

def read_requirements(filename: os.PathLike):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]


required_deps = read_requirements("requirements.txt")


def __createManifest__(subdirs):
    """inventory all files in path and create a manifest file"""
    current = os.path.dirname(__file__)
    relative_paths = [os.path.relpath(path, current) for path in subdirs]
    with open(os.path.join(current, "MANIFEST.in"), "w") as manifest:
        manifest.writelines(
            "recursive-include {} *.json".format(" ".join(relative_paths))
        )


add_il = os.path.join(os.path.dirname(__file__), "plmfit")

__createManifest__([add_il])
print("MANIFEST.in created")
setup(
    name="plmfit",
    version="1.0.0",
    description="PLMFit",
    long_description="PLMFit",
    author="Thomas Bikias, Evangelos Stamkopoulos",
    packages=find_packages(),
    license="MIT",
    zip_safe=True,
    install_requires=required_deps,
    include_package_data=True,
    entry_points={"console_scripts": ["plmfit=plmfit.__main__:main"]},
)
