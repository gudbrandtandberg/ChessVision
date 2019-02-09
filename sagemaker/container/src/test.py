import tensorflow as tf
import keras
from square_classifier import build_square_classifier
from u_net import get_unet_256
import os
import numpy as np
from keras import backend as K
from keras.models import load_model
import flask 

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

app = flask.Flask(__name__)
model_path = "model/"
graph1 = tf.Graph()
graph2 = tf.Graph()
with graph1.as_default():
    board_extractor = get_unet_256()
    board_extractor.load_weights(os.path.join(model_path, 'best_extractor.hdf5'))
with graph1.as_default():
    square_classifier = load_model(os.path.join(model_path, 'best_classifier.hdf5'))

@app.route("/", methods=["GET"])
def endpoint():
    data_256 = np.random.random((1, 256, 256, 3))
    data_512 = np.random.random((512, 512))

    global graph1, graph2

    with graph1.as_default():
        masks = board_extractor.predict(data_256)
    squares, _ = extract_squares(data_512)
    with graph1.as_default():
        res = square_classifier.predict(squares)

    print(res.shape)
    print("OK")

if __name__ == "__main__":
    app.run()
