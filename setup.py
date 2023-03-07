from setuptools import setup, find_packages
from sapicore import __version__

with open("README.md") as f:
    long_description = f.read()

setup(
    name="sapicore",
    version=__version__,
    description="A PyTorch-based framework for neuromorphic modeling. ",
    long_description=long_description,
    url="https://github.com/cplab/sapicore",
    author="Framework architecture by Matthew Einhorn and Roy Moyal. "
    "Engine infrastructure, simulator, and visualization tools by Roy Moyal. "
    "Neuromorphic algorithms by Ayon Borthakur, Roy Moyal, and Thomas Cleland. "
    "Tutorials and tutorials by Roy Moyal and Jeremy Forest. "
    "Project of the Computational Physiology Laboratory at Cornell University",
    author_email="rm875@cornell.edu, me263@cornell.edu, ab2535@cornell.edu, tac29@cornell.edu",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
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
        "natsort",
        "alive_progress",
        "tree-config",
    ],
    extras_require={
        "dev": ["pytest", "coverage", "flake8", "sphinx", "sphinx-rtd-theme"],
    },
)
