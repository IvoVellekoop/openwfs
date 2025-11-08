"""A wrapper for the setuptools build backend.

This wrapper adds some custom build steps:
- Extract the version number from pyproject.toml and store it in _version.py
- Extract the git revision from the repository and store it in _version.py
- Update README.md file by compiling it from the docs/source/readme.rst template (only if the source file has changed)
"""

import setuptools.build_meta as build_meta
import subprocess, pathlib



def pre_build(root_path):
    # build README.md if its source file (docs/source/readme.rst) has changed
    try:
        readme_modified = subprocess.check_output(["git", "diff", "source/readme.rst"], text=True, cwd=root_path / "docs").strip()
        if readme_modified:
            try:
                print("building README.md")
                subprocess.check_call([root_path / "docs" / "make.bat", "markdown"], cwd=root_path / "docs")
            except Exception as e:
                print(f"WARNING: failed to build README.md: {e}")
    except Exception as e:
        print(f"WARNING: failed to run git diff: {e}")


    # get git revision
    try:
        git_revision = subprocess.check_output(["git", "describe", "--tags"], text=True, cwd=root_path).strip()
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
    with open(root_path / "openwfs" / "_version.py", "w") as f:
        f.write('# This file is generated automatically by the build system (uv build, poetry build, see build_backend.py for details)\n')
        f.write(f'__version__ = "{version}"\n')
        f.write('"""Version number reported in the package metadata."""\n\n')
        f.write(f'__git_revision__ = "{git_revision}"\n')
        f.write('"""Git-reported information: Latest tag found in the code branch - number of commits since that tag - hash of current commit."""\n')


def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    root_path = pathlib.Path(wheel_directory).parent
    pre_build(root_path)
    return build_meta.build_wheel(wheel_directory, config_settings, metadata_directory)

def build_sdist(sdist_directory, config_settings=None):
    return build_meta.build_sdist(sdist_directory, config_settings)