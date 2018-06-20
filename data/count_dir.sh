#!/bin/bash
for d in B _b N _n R _r Q _q K _k f; do
    n=$(ls $1$d |wc -l)
    echo "$1$d $n"
    #cp $1$d/* squares_gen2/$d/
done
