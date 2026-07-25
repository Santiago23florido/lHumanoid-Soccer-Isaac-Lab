"""Setuptools entry point for Isaac Lab editable installs."""

from setuptools import find_packages, setup

setup(
    name="humanoid-soccer-lab",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
)
