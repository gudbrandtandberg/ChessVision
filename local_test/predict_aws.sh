#!/bin/bash

CHESSVISION_ENDPOINT = "https://4e4q3rd1b5.execute-api.eu-central-1.amazonaws.com/default/chessvisionClient"

curl -X POST \
-H "Content-Type: application/json" \
-d @test.json $CHESSVISION_ENDPOINT

