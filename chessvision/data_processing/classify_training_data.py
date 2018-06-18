"""Classifies alot of square images using the v1 NN, 
in order to generate more data for training NN v2."""

import cv2
from keras.models import load_model
from extract_squares import extract_squares
from square_classifier import build_square_classifier
import numpy as np
from util import listdir_nohidden
import os
import matplotlib.pyplot as plt

# python data_processing/classify_training_data.py

board_dir = "../data/new_boards/"
out_dir = "../data/squares_gen3/"
label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]
dir_names = ["R", "_r", "K", "_k", "Q", "_q", "N", "_n", "P", "_p", "B", "_b", "f"]
num_examples = {}
for lbl in label_names:
    num_examples[lbl] = 0

board_filenames = listdir_nohidden(board_dir)
board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

model = build_square_classifier()
model.load_weights('../weights/best_weights_square.hdf5')

for board_img, fname in zip(board_imgs, board_filenames):
    
    squares, names = extract_squares(board_img)

    X = squares.astype('float32')
    #X = X.reshape(X.shape[0], 64, 64, 1)
    X /= 255

    predictions = model.predict(X)
    predictions = np.argmax(predictions, axis=1)

    for pred, sq, name in zip(predictions, squares, names):
        num_examples[label_names[pred]] += 1
        dir_name = dir_names[pred]
        path = os.path.join(out_dir, dir_name)
        if not os.path.isdir(path):
            os.mkdir(path)
        filename = name + "_" + fname
        cv2.imwrite(os.path.join(path, filename), sq)
        
        #print(path, filename)
print(num_examples)

# last output: 
#{'n': 63, 'R': 111, 'b': 320, 'B': 57, 'P': 87, 'k': 497, 'N': 116, 'Q': 89, 'p': 404, 'r': 41, 'K': 126, 'q': 214, 'f': 2163}