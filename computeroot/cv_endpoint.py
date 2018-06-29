import os
from flask import Flask, flash, request, redirect, url_for, make_response, current_app
from werkzeug.utils import secure_filename
from datetime import timedelta
from functools import update_wrapper
import uuid
import json
import platform
import numpy as np
import base64 
import matplotlib.pyplot as plt

from chessvision import classify_raw
import cv_globals
import cv2
from stockfish import Stockfish
from extract_squares import extract_squares
from board_extractor import load_extractor
from board_classifier import load_classifier

from util import BoardExtractionError

app = Flask(__name__)

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


ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(cv_globals.CVROOT, "computeroot/user_uploads/")
app.config['TMP_FOLDER'] = os.path.join(cv_globals.CVROOT, "computeroot/tmp")

app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/cv_algo/', methods=['POST'])
@crossdomain(origin='*')
def predict_img():
    print("CV-Algo invoked")

    image = read_image_from_formdata()

    if image is not None:
        
        raw_id = str(uuid.uuid4())
        filename = secure_filename(raw_id) + ".JPG"
        tmp_loc = os.path.join(app.config['TMP_FOLDER'], filename)
        
        tmp_path = os.path.abspath(tmp_loc)
        cv2.imwrite(tmp_path, image)
        
        # check if input image is flipped (from black's perspective)
        flip = False
        if "flip" in request.form:
            flip = request.form["flip"] == "true"

        try:
            board_img, _, FEN, _ = classify_raw(image, filename, board_model, sq_model, flip=flip)
            #move file to success raw folder
            os.rename(tmp_loc, app.config["UPLOAD_FOLDER"], "raw", filename))
            cv2.imwrite("./user_uploads/boards/x_" + filename, board_img)
    
        except BoardExtractionError as e:
            #move file to success raw folder
            os.rename(tmp_loc, app.config["UPLOAD_FOLDER"], "raw", filename))
            return e.json_string()

        ret = '{{ "FEN": "{0}", "id": "{1}", "error": "false" }}'.format(FEN, raw_id)

        return ret
    
    return '{"error": "true", "errorMsg": "Fuck!"}'

piece2dir = {"wR": "R", "bR": "_r", "wK": "K", "bK": "_k", "wQ": "Q", "bQ": "_q",
 "wN": "N", "bN": "_n", "wP": "P", "bP": "_p", "wB": "B", "bB": "_b", "f": "f"}

@app.route('/feedback/', methods=['POST'])
@crossdomain(origin='*')
def receive_feedback():
    res = '{"success": "false"}'

    if "id" not in request.form or "position" not in request.form or "flip" not in request.form:
        print("Missing form data, abort!")
        return res

    raw_id = request.form['id']
    position = json.loads(request.form["position"])
    flip = request.form["flip"] == "true"

    board_filename = "x_" + raw_id + ".JPG"
    board_src = os.path.join(app.config["UPLOAD_FOLDER"], "boards/", board_filename)

    if not os.path.isfile(board_src):
        return res

    board = cv2.imread(board_src, 0)

    squares, names = extract_squares(board, flip=flip)

    # Save each square using the 'position' variable
    for sq, name in zip(squares, names):
        
        if name not in position:
            label = "f"
        else:
            label = position[name]
        
        fname = name + "_" + raw_id + ".JPG"
        out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "squares/", piece2dir[label])
        outfile = os.path.join(out_dir, fname)
        cv2.imwrite(outfile, sq)
        
    # remove the board file
    os.remove(board_src)

    return '{ "success": "true" }'

@app.route('/analyze/', methods=['POST'])
@crossdomain(origin='*')
def analyze():
    res = '{{ "success": "false" }}'

    if "FEN" not in request.form:
        print("No FEN in the form data...")
        return res

    fen = request.form["FEN"]
    print(fen)
    # check input is legal

    plat = platform.system()
    if plat == "Linux":
        sf_binary = "./stockfish-9-64-linux"
    elif plat == "Darwin":
        sf_binary = "./stockfish-9-64-mac"
    else:
        print("No support for windows..")
        return res

    stockfish = Stockfish(sf_binary, depth=10)
    try: 
        stockfish.set_fen_position(fen)
    except:
        print("BOARD ILLEGAL!")
        return res
    try:
        best_move = stockfish.get_best_move()
        print("Best move is: {}".format(best_move))
    except:
        print("STOCKFISH FAILED")
        return res
    
    return '{{ "success": "true", "bestMove": "{}" }}'.format(best_move)

def data_uri_to_cv2_img(uri):

    

    return img

def read_image_from_formdata():
    # check if the post request has the file part
    
    if 'file' not in request.files:
        print("No file")
        return None

    file = request.files['file']
    
    data = file.read()
    
    nparr = np.frombuffer(data, np.uint8)
    
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    file.close()
    
    return img


def load_models():
    global sq_model, board_model

    sq_model = load_classifier()
    board_model = load_extractor()

    return board_model, sq_model

if __name__ == '__main__':

    board_model, sq_model = load_models()
    app.run(host='127.0.0.1', port=7777)
