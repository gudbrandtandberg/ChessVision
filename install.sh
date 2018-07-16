#!/bin/bash
conda create --yes -n chessvision python=3.5
source activate chessvision
while read requirement; do conda install --yes $requirement; done < requirements.txt 2>error.log
conda install --channel https://conda.anaconda.org/menpo opencv3
pip install python-chess
pip install stockfish


# To use with the Intel Math Kernel Library
#export MKL_THREADING_LAYER=GNU