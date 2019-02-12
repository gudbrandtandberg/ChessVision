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
import logging 

app = Flask(__name__)

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

class RequestFormatter(logging.Formatter):
    def format(self, record):
        record.url = request.url
        record.remote_addr = request.remote_addr
        return super().format(record)

formatter = RequestFormatter(
    "[%(asctime)s] %(remote_addr)s requested %(url)s. "
    "%(levelname)s in %(name)s: %(message)s"
)

logger = logging.getLogger("chessvision")
file_handler = logging.FileHandler('cv_endpoint.log', 'w')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
<<<<<<< HEAD
logger.setLevel(logging.DEBUG)
=======
logger.setLevel(logging.INFO)
>>>>>>> 2388d53994ce1daaa0fc00abd947c5859faaa050

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
    logger.info("CV-Algo invoked")
    
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
            logger.info("Processing image {}".format(filename))
            board_img, _, _, _, FEN, _, _ = classify_raw(image, filename, board_model, sq_model, flip=flip)
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
            cv2.imwrite("./user_uploads/boards/x_" + filename, board_img)
    
        except BoardExtractionError as e:
            #move file to success raw folder
            os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
            return e.json_string()
<<<<<<< HEAD
        
        except FileNotFoundError as e:
            logger.debug("Unexpected error: {}".format(e))
            return '{{"error": "true"}}'
=======
>>>>>>> 2388d53994ce1daaa0fc00abd947c5859faaa050

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

piece2dir = {"R": "R", "r": "_r", "K": "K", "k": "_k", "Q": "Q", "q": "_q",
 "N": "N", "n": "_n", "P": "P", "p": "_p", "B": "B", "b": "_b", "f": "f"}

dict = {"wR": "R", "bR": "r", "wK": "K", "bK": "k", "wQ": "Q", "bQ": "q",
 "wN": "N", "bN": "n", "wP": "P", "bP": "p", "wB": "B", "bB": "b", "f": "f"}

def convertPosition(position):
    new = {}
    for key in position:
        new[key] = dict[position[key]]
    return new
    
def FEN2JSON(fen):
    piecemap = chess.Board(fen=fen).piece_map()
    predictedPos = {}
    for square_index in piecemap:
        square = chess.SQUARE_NAMES[square_index]
        predictedPos[square] = str(piecemap[square_index].symbol())
    return predictedPos

@app.route('/feedback/', methods=['POST'])
@crossdomain(origin='*')
def receive_feedback():
    res = '{"success": "false"}'

    if "id" not in request.form or "position" not in request.form or "flip" not in request.form:
        logger.error("Missing form data, abort!")
        return res

    raw_id = request.form['id']
    position = json.loads(request.form["position"])
    flip = request.form["flip"] == "true"
    predictedFEN = request.form["predictedFEN"]
    predictedPos = FEN2JSON(predictedFEN)
    position = convertPosition(position)
    board_filename = "x_" + raw_id + ".JPG"
    board_filename = os.path.join(app.config["UPLOAD_FOLDER"], "boards/", board_filename)

    if not os.path.isfile(board_filename):
        return res

    board = cv2.imread(board_filename, 0)
    squares, names = extract_squares(board, flip=flip)

    # Save each square using the 'position' dictionary from chessboard.js
    for sq, name in zip(squares, names):
        if name not in position:
            label = "f"
        else:
            label = position[name]

        if name not in predictedPos:
            predictedLabel = "f"
        else:
            predictedLabel = predictedPos[name]
        
        if predictedLabel == label:
            continue
        
        # Else, the prediction was incorrect, save it to learn from it later
        fname = str(uuid.uuid4()) + ".JPG"
        out_dir = os.path.join(app.config["UPLOAD_FOLDER"], "squares/", piece2dir[label])
        outfile = os.path.join(out_dir, fname)
        cv2.imwrite(outfile, sq)

    # remove the board file
    os.remove(board_filename)

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
        sf_binary = os.path.join(cv_globals.compute_root, "stockfish-9-64-linux")
    elif plat == "Darwin":
        sf_binary = os.path.join(cv_globals.compute_root, "stockfish-9-64-mac")
    else:
        logger.error("No support for windows..")
        return res

    stockfish = Engine(sf_binary, depth=10)

    try: 
        stockfish.setposition(fen)
    except Exception as e:
        logger.error("BOARD ILLEGAL!: {}".format(e))
        return res
    try:
        best_move = stockfish.bestmove()
        logger.debug("Best move is: {}".format(best_move["bestmove"])) #is '(none)' if there is mate
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
        logger.error("STOCKFISH FAILED: {}".format(e))
        return res
    
    return {"best_move": best_move["bestmove"], "score": score, "mate": mate}

@app.route('/analyze/', methods=['POST'])
@crossdomain(origin='*')
def engine_analyze():
    logger.info("Analyzing position using Stockfish")
    res = '{{ "success": "false" }}'

    if "FEN" not in request.form:
        logger.error("No FEN in the form data...")
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
        logger.error("No file")
        return None

    try: 
        file = request.files['file']
        data = file.read()
        nparr = np.frombuffer(data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        file.close()
    except: 
        img = None

    return img


def load_models():
    #global sq_model, board_model

    sq_model = load_classifier(weights=cv_globals.square_weights)
    board_model = load_extractor(weights=cv_globals.board_weights)

    return board_model, sq_model

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--local", action="store_true")
    args = parser.parse_args()

    port = 7777 if args.local else 8080
    board_model, sq_model = load_models()
    
    app.run(host='0.0.0.0', port=port)