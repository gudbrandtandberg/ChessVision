#!/bin/bash
conda create --yes -n chessvision python=3.5
source activate chessvision
while read requirement; do conda install --yes $requirement; done < requirements.txt 2>error.log
pip install python-chess
pip install stockfish
