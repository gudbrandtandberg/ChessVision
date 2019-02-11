#!/bin/bash

(echo -n '{"flip": "false", "image": "'; base64 test_img.JPG; echo '"}') | curl -H "Content-Type: application/json" -d @-  http://127.0.0.1:8080/invocations

