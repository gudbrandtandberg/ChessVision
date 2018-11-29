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
import argparse

from chessvision import classify_raw
import cv_globals
import cv2
import chess
from stockfishpy.stockfishpy import *
from extract_squares import extract_squares
from u_net import load_extractor
from square_classifier import load_classifier
from util import BoardExtractionError
from flask.logging import default_handler

app = Flask(__name__)
#app.logger.removeHandler(default_handler)
#from logging.config import dictConfig
#import logging
#import logging.config

#logging.config.fileConfig("cv.log")

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

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(cv_globals.CVROOT, "computeroot/user_uploads/")
app.config['TMP_FOLDER'] = os.path.join(cv_globals.CVROOT, "computeroot/tmp/")
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app.secret_key = 'super secret key'

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/cv_algo/', methods=['POST'])
@crossdomain(origin='*')
def predict_img():
    app.logger.info("CV-Algo invoked")
    
    #Host, User-Agent, Content-Length, Origin
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
        if "tomove" in request.form:
            tomove = request.form["tomove"]

        try:
            board_img, _, _, _, FEN, _ = classify_raw(image, filename, board_model, sq_model, flip=flip)
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
            cv2.imwrite("./user_uploads/boards/x_" + filename, board_img)
    
        except BoardExtractionError as e:
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
            return e.json_string()

        FEN = expandFEN(FEN, tomove)
        
        analysis = analyze(FEN, tomove)
        if "error" in analysis:
            score, mate = "None", "None"
        else:
            score, mate = analysis["score"], analysis["mate"]
        
        ret = '{{ "FEN": "{}", "id": "{}", "error": "false", "score": "{}", "mate": "{}" }}'.format(FEN, raw_id, score, mate)

        return ret
    
    return '{"error": "true", "errorMsg": "Oops!!"}'

def expandFEN(FEN, tomove):
    return "{} {} - - 0 1".format(FEN, tomove)

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

    # Save each square using the 'position' dictionary from chessboard.js
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

def analyze(fen, move):
    # check input is legal!!
    res = {"error": True}

    board = chess.Board(fen=fen)
    if not board.is_valid():
        return res
    
    if board.is_game_over():
        return res

    plat = platform.system()
    if plat == "Linux":
        sf_binary = "./stockfish-9-64-linux"
    elif plat == "Darwin":
        sf_binary = "./stockfish-9-64-mac"
    else:
        print("No support for windows..")
        return res

    stockfish = Engine(sf_binary, depth=10)

    try: 
        stockfish.setposition(fen)
    except Exception as e:
        print("BOARD ILLEGAL!: {}".format(e))
        return res
    try:
        best_move = stockfish.bestmove()
        print("Best move is: {}".format(best_move["bestmove"])) #is '(none)' if there is mate
        info = best_move["info"].split()

        if "cp" in info:
            score = float(info[info.index("cp")+1]) / 100.
            if move == "b": 
                score *= -1
            mate = ""
        elif "mate" in info:
            score = "None"
            mate = int(info[info.index("mate")+1])
    
    except Exception as e:
        print("STOCKFISH FAILED: {}".format(e))
        return res
    
    return {"best_move": best_move["bestmove"], "score": score, "mate": mate}

@app.route('/analyze/', methods=['POST'])
@crossdomain(origin='*')
def engine_analyze():
    print("Analyzing position using Stockfish")
    res = '{{ "success": "false" }}'

    if "FEN" not in request.form:
        print("No FEN in the form data...")
        return res

    fen = request.form["FEN"]
    move = fen.split()[1]
    
    analysis = analyze(fen, move)
    if "error" in analysis: 
        return res

    best_move, score, mate = analysis["best_move"], analysis["score"], analysis["mate"]
    
    return '{{ "success": "true", "bestMove": "{}", "score": "{}", "mate": "{}" }}'.format(best_move, score, mate)

@app.route('/ping/', methods=['GET'])
@crossdomain(origin='*')
def ping():
    return '{{ "success": "true"}}'


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
    #global sq_model, board_model

    sq_model = load_classifier(weights=cv_globals.square_weights)
    board_model = load_extractor()

    return board_model, sq_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", type=bool, default=False)
    args = parser.parse_args()
    
    port = 8080 if args.local else 7777
    board_model, sq_model = load_models()
    app.run(host='0.0.0.0', port=port)