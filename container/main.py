import base64
import json
import logging
import os
import uuid
from datetime import date, timedelta
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

logger = logging.getLogger("chessvision")
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())

s3Client = boto3.client('s3')

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

@app.before_first_request
def load_models():
    logger.info("Loading models...")
    model_path = "/weights"
    board_extractor = get_unet_256()
    board_extractor.load_weights(os.path.join(model_path, 'best_extractor.hdf5'))
    square_classifier = load_model(os.path.join(model_path, 'best_classifier.hdf5'))
    current_app.board_extractor = board_extractor
    current_app.square_classifier = square_classifier
    logger.info("Models loaded")

@app.route('/ping', methods=['GET'])
def ping():
    """
    Healthcheck.
    """
    logger.debug("Pinged")
    board_extractor = current_app.board_extractor
    square_classifier = current_app.square_classifier
    error = ""
    if not square_classifier or not board_extractor:
        error = "models failed to load"
    status = not error
    return flask.Response(
        response=json.dumps({"status": status, "error": error}),
        status=status,
        mimetype='application/json')

@app.route('/invocations', methods=['POST', "GET", "OPTIONS"])
@crossdomain(origin='*')
def chessvision_algo():
    """
    """
    logger.info("Chessvision algorithm invoked")
    board_extractor = current_app.board_extractor
    square_classifier = current_app.square_classifier
    if not square_classifier or not board_extractor:
        raise Exception("Models not loaded in this application")

    img = None
    flipped = None

    if flask.request.content_type != 'application/json':
        raise Exception("This endpoints only accepts content-type application/json")

    try:
        data = json.loads(flask.request.data.decode('utf-8'))
        flipped = data["flip"] == "true"
        img = read_image_from_b64(data["image"])
    except Exception as e:
        return flask.Response(
            response=f"Failed to read image from payload: {e}",
            status=415,
            mimetype="application/json")

    
    # Upload image to S3
    filename = str(uuid.uuid4()) + ".JPG"
    date_prefix = f"{date.today().year}/{date.today().month}/{date.today().day}/"
    enc = cv2.imencode(".jpg", img)[1].tobytes()
    s3Client.put_object(
        Body=enc,
        Bucket="chessvision-bucket",
        Key="raw-uploads/" + date_prefix + filename,
        ContentType="image/jpeg"
        )

    # Classify image
    try:
        _, _, _, _, FEN, _, _ = classify_raw(img, filename, board_extractor, square_classifier, flip=flipped)
    except BoardExtractionError as e:
        return flask.Response(
            response=json.dumps({"error": f"ChessVision algorithm failed with error '{e}'"}),
            status=500,
            mimetype="application/json")

    result = {"FEN": FEN, "error": ""}
    return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")

if __name__ == "__main__":
    print("Running debug server")
    app.run(host='0.0.0.0', port=8080)
