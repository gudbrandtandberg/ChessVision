"""Classifies alot of square images using the v1 NN, 
in order to generate more data for training NN v2."""

import cv2
from extract_squares import extract_squares
from board_classifier import load_classifier
import numpy as np
from util import listdir_nohidden
import os

# python data_processing/classify_training_data.py

board_dir = "../data/new_boards/"
out_dir = "../data/new_squares/"
label_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']            
dir_names = ["B", "K", "N", "P", "Q", "R", "_b", "_k", "_n", "_p", "_q", "_r", "f"]

for d in dir_names:
    if not os.path.isdir(out_dir + d):
        os.mkdir(out_dir + d)

num_examples = {}

for lbl in label_names:
    num_examples[lbl] = 0

board_filenames = listdir_nohidden(board_dir)
board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

model = load_classifier()

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