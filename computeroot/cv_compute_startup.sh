#!/bin/bash
echo "Running custom startup script"
source activate chessvision
cd ~/ChessVision/computeroot
echo "Running cv_endpoint.py"
python "$CVROOT"computeroot/cv_endpoint.py
