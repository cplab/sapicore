#!/bin/bash
echo ========== deleting book files ==========
rm -r book/_build/
rm -r book/docs/

echo ========== regenerating docs ==========
./refresh.sh

echo  ========== symlink README ==========
ln -s ../../README.md book/README.md

echo  ========== building book ==========
# jupyter-book build book/

# This is a workaround to build the jupyterbook with a specified path outside the /book path so that
# sphinx autodoc and autosummary can pull the docstrings from the code.
jupyter-book config sphinx book/

echo "import os" >> book/conf.py
echo "import sys" >> book/conf.py
echo "sys.path.insert(0, os.path.abspath('../../'))" >> book/conf.py
echo "sys.path.insert(0, os.path.abspath('../../sapicore/'))" >> book/conf.py

sphinx-build book/ book/_build/html/ -b html
