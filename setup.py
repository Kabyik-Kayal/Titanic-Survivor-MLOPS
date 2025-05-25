"""Setup script for the Titanic Survival MLOPS System package."""
from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="Titanic Survival MLOPS",
    version="0.1",
    author="Kabyik",
    packages=find_packages(),
    install_requires = requirements,
)