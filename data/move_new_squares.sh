#!/bin/bash
for d in B N K Q R _b _n _k _q _r; do
    n=$(ls $1$d |wc -l)
    echo "$1$d $n"
    #cp $1$d/* squares_gen2/$d/
done
