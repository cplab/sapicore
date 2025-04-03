from setuptools import setup, find_packages
from sapicore import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sapicore",
    version=__version__,
    description="A framework for spiking neural network modeling. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cplab/sapicore",
    author="Roy Moyal, Matthew Einhorn, Ayon Borthakur, Thomas Cleland",
    author_email="rm875@cornell.edu, me263@cornell.edu, ayon.borthakur@ai.iith.ac.in, " "tac29@cornell.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(),
    install_requires=[
        "h5py",
        "numpy",
        "scipy",
        "pandas",
        "torch",
        "tensorboard",
        "networkx",
        "scikit-learn",
        "matplotlib",
        "PyYAML",
        "nixio",
        "dill",
        "natsort",
        "alive_progress",
        "tree-config",
        "pytest",
        "setuptools",
    ],
    extras_require={
        "dev": ["coverage", "flake8", "sphinx<7.0.0", "sphinx-rtd-theme", "m2r2"],
    },
)
