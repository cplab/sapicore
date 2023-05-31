#!/bin/bash
if [ -d "source/api/_autosummary" ]; then
    echo Removing existing autosummary folder
    rm -r source/api/_autosummary
fi

make clean
make html
# make latexpdf source build/pdf
