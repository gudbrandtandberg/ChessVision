#!/bin/bash

#Usage: ./move_new_squares.sh new_squares/
out_dir="squares/all/"
for d in B _b N _n R _r Q _q K _k f P _p; do
    n=$(ls $1$d |wc -l)
    echo "Moving $n new $d's..."
    #cp $1$d/* $out_dir$d/
done
