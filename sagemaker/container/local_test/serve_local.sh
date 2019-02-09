#!/bin/sh

image=$1

docker run -v $(pwd)/test_dir:/opt/ml -p 8080:8080 --rm ${image} serve

# docker build -t chessvision-algo .

# docker run -p 8080:8080 chessvision-algo serve