import base64
import json
import os
import uuid
from datetime import timedelta
from functools import update_wrapper

import boto3
import cv2
import flask
import numpy as np
from cv2 import IMREAD_COLOR, imdecode
from flask import current_app, make_response, request
from tensorflow.keras.models import load_model

from chessvision.chessvision import classify_raw
from chessvision.model.u_net import get_unet_256
from chessvision.util import BoardExtractionError

model_path = "/weights"

client = boto3.client('s3')

print("Loading models...")
board_extractor = get_unet_256()
board_extractor.load_weights(os.path.join(model_path, 'best_extractor.hdf5'))

square_classifier = load_model(os.path.join(model_path, 'best_classifier.hdf5'))
print("Loading models... DONE!")


def crossdomain(origin=None, methods=None, headers=None, max_age=21600,
                attach_to_all=True, automatic_options=True):
    """Decorator function that allows crossdomain requests.
      Courtesy of
      https://blog.skyred.fi/articles/better-crossdomain-snippet-for-flask.html
    """
    if methods is not None:
        methods = ', '.join(sorted(x.upper() for x in methods))
    if headers is not None and not isinstance(headers, str):
        headers = ', '.join(x.upper() for x in headers)
    if not isinstance(origin, str):
        origin = ', '.join(origin)
    if isinstance(max_age, timedelta):
        max_age = max_age.total_seconds()

    def get_methods():
        """ Determines which methods are allowed
        """
        if methods is not None:
            return methods

        options_resp = current_app.make_default_options_response()
        return options_resp.headers['allow']

    def decorator(f):
        """The decorator function
        """
        def wrapped_function(*args, **kwargs):
            """Caries out the actual cross domain code
            """
            if automatic_options and request.method == 'OPTIONS':
                resp = current_app.make_default_options_response()
            else:
                resp = make_response(f(*args, **kwargs))
            if not attach_to_all and request.method != 'OPTIONS':
                return resp

            h = resp.headers
            h['Access-Control-Allow-Origin'] = origin
            h['Access-Control-Allow-Methods'] = get_methods()
            h['Access-Control-Max-Age'] = str(max_age)
            h['Access-Control-Allow-Credentials'] = 'true'
            h['Access-Control-Allow-Headers'] = \
                "Origin, X-Requested-With, Content-Type, Accept, Authorization"
            if headers is not None:
                h['Access-Control-Allow-Headers'] = headers
            return resp

        f.provide_automatic_options = False
        return update_wrapper(wrapped_function, f)
    return decorator

def read_image_from_b64(b64string):
    buffer = base64.b64decode(b64string)
    nparr = np.frombuffer(buffer, dtype=np.uint8)
    img = imdecode(nparr, IMREAD_COLOR)
    return img

# The flask app for serving predictions
app = flask.Flask(__name__)

@app.route('/ping', methods=['GET'])
def ping():
    """
    Determine if the container is working and healthy.
    In this sample container, we declare it healthy if we can load the model
    successfully.
    """
    print("Pinged")
    health = True
    status = 200 if health else 404
    return flask.Response(
        response="Yeah, bro!",
        status=status,
        mimetype='application/json')

@app.route('/invocations', methods=['POST', "GET", "OPTIONS"])
@crossdomain(origin='*')
def chessvision_algo():
    """
    """
    print("Invoked")
    img = None
    flipped = None

    if flask.request.content_type == 'application/json':
        print("Got data")
        data = json.loads(flask.request.data.decode('utf-8'))
        flipped = data["flip"] == "true"
        img = read_image_from_b64(data["image"])

    if img is None or flipped is None:
        print("Did not got data")
        return flask.Response(
            response="Could not parse input!",
            status=415,
            mimetype="application/json")

    # Upload image to S3
    filename = str(uuid.uuid4()) + ".JPG"
    cv2.imwrite(filename, img)
    with open(filename, "rb") as data:
        client.upload_fileobj(data, "chessvision-bucket", "raw-uploads/" + filename)
    os.remove(filename)

    try:
        _, _, _, _, FEN, _, _ = classify_raw(img, filename, board_extractor, square_classifier, flip=flipped)
    
    except BoardExtractionError:
        return flask.Response(
            response="ChessVision algorithm failed",
            status=500,
            mimetype="application/json")

    result = {"FEN": FEN}

    return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")

if __name__ == "__main__":
    print("Running server")
    app.run(host='0.0.0.0', port=8080)
