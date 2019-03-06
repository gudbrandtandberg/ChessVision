#!/bin/bash

CHESSVISION_ENDPOINT = "http://127.0.0.1:8080/invocations"

curl -X POST \
-H "Content-Type: application/json" \
-d @test.json $CHESSVISION_ENDPOINT