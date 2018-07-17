#!/bin/bash

mkdir -p user_uploads/

mkdir -p user_uploads/boards/
mkdir -p user_uploads/squares/
mkdir -p user_uploads/raw/

mkdir -p tmp

for d in B _b N _n R _r Q _q K _k f P _p; do
	mkdir -p "user_uploads/squares/$d"
done
