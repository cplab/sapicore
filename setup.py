from setuptools import setup, find_packages
from sapicore import __version__

with open("README.rst") as f:
    long_description = f.read()

setup(
    name="sapicore",
    version=__version__,
    description="A PyTorch-based framework for neuromorphic modeling",
    long_description=long_description,
    url="https://github.com/cplab/sapicore",
    author="Neuromorphic algorithms by Ayon Borthakur and Thomas Cleland, "
           "Framework architecture by Matthew Einhorn. Project of the "
           "Computational Physiology Laboratory at Cornell University",
    author_email="ab2535@cornell.edu",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "tqdm",
        "pandas",
        'ruamel.yaml',
        'nixio',
        'tree-config',
    ],
    extras_require={
        'dev': [
            'pytest>=3.6', 'pytest-cov', 'flake8', 'sphinx-rtd-theme',
            'coveralls', 'sphinx'],
    },
)
