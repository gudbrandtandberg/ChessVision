from __future__ import print_function

import os
import flask
import json
import base64
from cv2 import resize, cvtColor, flip, approxPolyDP, imdecode, findContours, arcLength, getPerspectiveTransform, warpPerspective, contourArea, boundingRect, COLOR_BGR2GRAY, IMREAD_COLOR, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS
import numpy as np
from u_net import get_unet_256
from square_classifier import build_square_classifier
import chessvision
import tensorflow as tf
from tensorflow import Graph
import chess
import cv2
from keras.models import load_model

prefix = 'weights/'
prefix = 'model/'
model_path = prefix
#model_path = os.path.join(prefix, 'model')

print("Loading models...")
board_extractor = get_unet_256()
board_extractor.load_weights(os.path.join(model_path, 'best_extractor.hdf5'))

square_classifier = load_model(os.path.join(model_path, 'best_classifier.hdf5'))
print("Loading models... DONE!")

graph = tf.get_default_graph()

# The flask app for serving predictions
app = flask.Flask(__name__)

def extract_squares(board, flip=False):
    ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    files = ["1", "2", "3", "4", "5", "6", "7", "8"]
    
    if flip:
        ranks = list(reversed(ranks))
        files = list(reversed(files))

    squares = []
    names = []
    ww, hh = board.shape
    w = int(ww / 8)
    h = int(hh / 8)

    for i in range(8):
        for j in range(8):
            squares.append(board[i*w:(i+1)*w, j*h:(j+1)*h])
            names.append(ranks[j]+files[7-i])
    
    squares = np.array(squares)
    squares = squares.reshape(squares.shape[0], 64, 64, 1)
    
    return squares, names


def read_image_from_b64(b64string):
    buffer = base64.b64decode(b64string)
    nparr = np.frombuffer(buffer, dtype=np.uint8)
    img = imdecode(nparr, IMREAD_COLOR)
    return img

def fix_mask(mask, threshold=80):
    # max_val = np.max(np.max(mask))
    # mask /= max_val
    mask *= 255
    mask = mask.astype(np.uint8)
    mask[mask > threshold] = 255
    mask[mask <= threshold] = 0
    return mask

def scale_approx(approx, orig_size):
    sf = orig_size[0]/256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


def rotate_quadrangle(approx):
    if approx[0,0,0] < approx[2,0,0]:
        approx = approx[[3, 0, 1, 2],:,:]
    return approx

def find_quadrangle(mask):

    contours, _ = findContours(mask, RETR_CCOMP, CHAIN_APPROX_TC89_KCOS)
    #first _
    if len(contours) > 1:
        #print("Found {} contour(s)".format(len(contours)))
        contours = ignore_contours(mask.shape, contours)
        #print("Filtered to {} contour(s)".format(len(contours)))

    if len(contours) == 0:
        return None
    
    approx = None

    # try to approximate and hope for a quad
    for i in range(len(contours)):
        cnt = contours[i]
        
        arclen = arcLength(cnt, True)
        candidate = approxPolyDP(cnt, 0.1*arclen, True)
        
        if len(candidate) != 4:
            continue

        approx = rotate_quadrangle(candidate)
        break

    return approx

def extract_perspective(image, approx, out_size):

    w, h = out_size[0], out_size[1]
    
    dest = ((0,0), (w, 0), (w,h), (0, h))

    approx = np.array(approx, np.float32)
    dest = np.array(dest, np.float32)

    coeffs = getPerspectiveTransform(approx, dest)

    return warpPerspective(image, coeffs, out_size)

def ignore_contours(img_shape,
                   contours,
                   min_ratio_bounding=0.6,
                   min_area_percentage=0.35,
                   max_area_percentage=1.0):

    ret = []
    mask_area = float(img_shape[0]*img_shape[1])

    for i in range(len(contours)):
        ca = contourArea(contours[i])
        ca /= mask_area
        if ca < min_area_percentage or ca > max_area_percentage:
            continue        
        _, _, w, h = tmp = boundingRect(contours[i])
        if ratio(h,w) < min_ratio_bounding:
            continue
        ret.append(contours[i])

    return ret

def ratio(a,b):
    if a == 0 or b == 0:
        return -1
    return min(a,b)/float(max(a,b))

def classification_logic(probs, names):
    
    initial_predictions = np.argmax(probs, axis=1)

    label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

    pred_labels = [label_names[p] for p in initial_predictions]

    # pred_labels = check_multiple_kings(pred_labels, probs)
    # pred_labels = check_bishops(pred_labels, probs, names)
    # pred_labels = check_pawns_not_on_first_rank(pred_labels, probs, names)
    
    board = build_board_from_labels(pred_labels, names)
    
    return board

def build_board_from_labels(labels, names):
    board = chess.BaseBoard(board_fen=None)
    for pred_label, sq in zip(labels, names):
        if pred_label == "f":
            piece = None
        else:
            piece = chess.Piece.from_symbol(pred_label)
        
        square = chess.SQUARE_NAMES.index(sq)
        board.set_piece_at(square, piece, promoted=False)
    return board

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

@app.route('/invocations', methods=['POST', "GET"])
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

    else:
        print("Did not got data")
        return flask.Response(
            response="Could not parse input!",
            status=415,
            mimetype="application/json")

    # Predict
    
    comp_img = resize(img, (256, 256))
   
    global graph
    mask = None
    img_batch = np.array([comp_img], np.float32) / 255
    
    with graph.as_default():
        mask = board_extractor.predict(img_batch)
    
    mask = mask[0].reshape((256, 256))
    mask = fix_mask(mask)
    
    approx = find_quadrangle(mask)

    if approx is None:
        print("Contour approximation failed!")
        return flask.Response(
            response="ChessVision algorithm failed!",
            status=500,
            mimetype="application/json")

    approx = scale_approx(approx, (img.shape[0], img.shape[1])) 
    board = extract_perspective(img, approx, (512, 512))
    board = cvtColor(board, COLOR_BGR2GRAY)
    board = flip(board, 1) # TODO: permute approximation instead..
    squares, names = extract_squares(board, flip=flipped)
    
    with graph.as_default():
        predictions = square_classifier.predict(squares)

    chessboard = classification_logic(predictions, names)
    FEN = chessboard.board_fen(promoted=False)
    result = {"FEN": FEN}

    return flask.Response(response=json.dumps(result), status=200, mimetype="application/json")

if __name__ == "__main__":
    print("Running server")
    app.run(host='0.0.0.0', port=8080)
