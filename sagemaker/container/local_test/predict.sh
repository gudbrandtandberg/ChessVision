#!/bin/bash

#payload=$1
#content=${2:-text/plain}

#curl --data-binary @${payload} -H "Content-Type: ${content}" -v http://localhost:5000/invocations

(echo -n '{"flip": "false", "image": "'; base64 test_img.JPG; echo '"}') | curl -H "Content-Type: application/json" -d @-  http://127.0.0.1:8080/invocations


#curl -H "Content-Type: application/json" -d '{"image": "heihei"}' http://0.0.0.0:5000/invocations
