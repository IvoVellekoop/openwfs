"""A wrapper for the setuptools build backend.

This wrapper adds some custom build steps:
- Extract the version number from pyproject.toml and store it in _version.py
- Extract the git revision from the repository and store it in _version.py
- Update README.md file by compiling it from the docs/source/readme.rst template (only if the source file has changed)
"""

import setuptools.build_meta as build_meta
import subprocess
from pathlib import Path


def pre_build(output_path):
    # Note: we overwrite README.md and _version.py in the original source folder
    # we don't use the output_path that is provided by the build backend.
    root_path = Path.cwd()
    output_path = Path.cwd()
    # build README.md if its source file (docs/source/readme.rst) has changed
    try:
        readme_modified = subprocess.check_output(
            ["git", "diff", "source/readme.rst"], text=True, cwd=str(root_path / "docs")
        ).strip()
        if readme_modified:
            try:
                print("building README.md")
                subprocess.check_call([str(root_path / "docs" / "make.bat"), "markdown"], cwd=str(root_path / "docs"))
            except Exception as e:
                print(f"WARNING: failed to build README.md: {e}")
        else:
            print("README.md is up to date")
    except Exception as e:
        print(f"WARNING: failed to run git diff: {e}")

    # get git revision. run in current working directory = location of pyproject.toml
    try:
        git_revision = subprocess.check_output(["git", "describe", "--tags"], text=True, cwd=str(root_path)).strip()
    except Exception as e:
        git_revision = f"unknown ({e})"

    # get version from pyproject.toml
    try:
        version = "unknown (not specified in pyproject.toml)"
        with open(root_path / "pyproject.toml") as f:
            for line in f.readlines():
                if line.startswith("version"):
                    version = line.split("=")[1].strip().strip('"')
                    break
    except Exception as e:
        version = f"unknown ({e})"

    # write version file
    print(f"version {version}")
    try:
        version_file = Path(output_path) / "openwfs" / "_version.py"
        print(f"writing version file {version_file}")
        with open(version_file, "w") as f:
            f.write(
                "# This file is generated automatically by the build system (uv build, poetry build, see build_backend.py for details)\n"
            )
            f.write(f'__version__ = "{version}"\n')
            f.write('"""Version number reported in the package metadata."""\n\n')
            f.write(f'__git_revision__ = "{git_revision}"\n')
            f.write(
                '"""Git-reported information: Latest tag found in the code branch - number of commits since that tag - hash of current commit."""\n'
            )
    except Exception as e:
        print(e)


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)


def build_sdist(sdist_directory, config_settings=None):
    pre_build(sdist_directory)
    return build_meta.build_sdist(sdist_directory, config_settings)
