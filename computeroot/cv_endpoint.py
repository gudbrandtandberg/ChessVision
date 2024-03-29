import argparse
import base64
import json
import logging
import os
import uuid
from datetime import timedelta
from functools import update_wrapper

import chess
import cv2
import flask
import numpy as np
from flask import Flask, current_app, make_response, request
from werkzeug.utils import secure_filename

import chessvision.cv_globals as cv_globals
from chessvision import classify_raw
from chessvision.data_processing.extract_squares import extract_squares
from chessvision.model.square_classifier import load_classifier
from chessvision.model.u_net import load_extractor
from chessvision.util import BoardExtractionError

app = Flask(__name__)

#log = logging.getLogger('werkzeug')
#log.setLevel(logging.ERROR)

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
file_handler = logging.FileHandler("logs/cv_endpoint.log", "w")
file_handler.setFormatter(formatter)
stream_handler = logging.StreamHandler()
logger.addHandler(stream_handler)
logger.addHandler(file_handler)
logger.setLevel(logging.DEBUG)

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

def read_image_from_b64(b64string):
    buffer = base64.b64decode(b64string)
    nparr = np.frombuffer(buffer, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

@app.route('/cv_algo/', methods=["POST", "OPTIONS"])
@crossdomain(origin='*')
def predict_img():
    logger.info("CV-Algo invoked")
    
    #Host, User-Agent, Content-Length, Origin
    if flask.request.content_type == 'application/json':
        print("Got data")
        data = json.loads(flask.request.data.decode('utf-8'))
        flipped = data["flip"] == "true"
        image = read_image_from_b64(data["image"])

    if image is None or flipped is None:
        print("Did not got data")
        return flask.Response(
            response="Could not parse input!",
            status=415,
            mimetype="application/json")
        
    raw_id = str(uuid.uuid4())
    filename = secure_filename(raw_id) + ".JPG"
    tmp_loc = os.path.join(app.config['TMP_FOLDER'], filename)
    
    tmp_path = os.path.abspath(tmp_loc)
    cv2.imwrite(tmp_path, image)

    try:
        logger.info("Processing image {}".format(filename))
        # global graph
        # with graph.as_default():
        board_img, _, _, _, FEN, _, _ = classify_raw(image, filename, board_model, sq_model, flip=flipped)
        
        #move file to success raw folder
        os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
        cv2.imwrite(os.path.join(app.config["UPLOAD_FOLDER"], "boards/x_" + filename), board_img)

    except BoardExtractionError as e:
        #move file to success raw folder
        os.rename(tmp_loc, os.path.join(app.config["UPLOAD_FOLDER"], "raw", filename))
        return e.json_string()
    
    except FileNotFoundError as e:
        logger.debug("Unexpected error: {}".format(e))
        return '{{"error": "true"}}'
    
    ret = '{{ "FEN": "{}", "id": "{}", "error": "false"}}'.format(FEN, raw_id)
    return ret

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

    data = json.loads(flask.request.data.decode('utf-8'))

    if "id" not in data or "position" not in data or "flip" not in data:
        logger.error("Missing form data, abort!")
        return res

    raw_id = data["id"]
    position = json.loads(data["position"])
    flip = data["flip"] == "true"
    predictedFEN = data["predictedFEN"]
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

# def analyze(fen, move):
#     # check input is legal!!
#     res = {"error": True}

#     board = chess.Board(fen=fen)
#     if not board.is_valid():
#         return res
    
#     if board.is_game_over():
#         return res

#     plat = platform.system()
#     if plat == "Linux":
#         sf_binary = os.path.join(cv_globals.compute_root, "stockfish-9-64-linux")
#     elif plat == "Darwin":
#         sf_binary = os.path.join(cv_globals.compute_root, "stockfish-9-64-mac")
#     else:
#         logger.error("No support for windows..")
#         return res

#     stockfish = Engine(sf_binary, depth=10)

#     try: 
#         stockfish.setposition(fen)
#     except Exception as e:
#         logger.error("BOARD ILLEGAL!: {}".format(e))
#         return res
#     try:
#         best_move = stockfish.bestmove()
#         logger.debug("Best move is: {}".format(best_move["bestmove"])) #is '(none)' if there is mate
#         info = best_move["info"].split()

#         if "cp" in info:
#             score = float(info[info.index("cp")+1]) / 100.
#             if move == "b": 
#                 score *= -1
#             mate = ""
#         elif "mate" in info:
#             score = "None"
#             mate = int(info[info.index("mate")+1])
    
#     except Exception as e:
#         logger.error("STOCKFISH FAILED: {}".format(e))
#         return res
    
#     return {"best_move": best_move["bestmove"], "score": score, "mate": mate}

# @app.route('/analyze/', methods=['POST'])
# @crossdomain(origin='*')
# def engine_analyze():
#     logger.info("Analyzing position using Stockfish")
#     res = '{{ "success": "false" }}'

#     if "FEN" not in request.form:
#         logger.error("No FEN in the form data...")
#         return res

#     fen = request.form["FEN"]
#     move = fen.split()[1]
    
#     analysis = analyze(fen, move)
#     if "error" in analysis: 
#         return res

#     best_move, score, mate = analysis["best_move"], analysis["score"], analysis["mate"]
    
#     return '{{ "success": "true", "bestMove": "{}", "score": "{}", "mate": "{}" }}'.format(best_move, score, mate)

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