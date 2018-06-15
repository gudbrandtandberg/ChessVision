"""Classifies alot of square images using the v1 NN, 
in order to generate more data for training NN v2."""

import cv2
from keras.models import load_model
from extract_squares import extract_squares
import numpy as np
from util import listdir_nohidden

board_dir = "../data/Segmentation/boards/"
out_dir = "../data/extra_training_data/"
label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]
dir_names = ["R", "_r", "K", "_k", "Q", "_q", "N", "_n", "P", "_p", "B", "_b", "f"]


board_filenames = listdir_nohidden(board_dir)
board_filenames = [b for b in board_filenames]
board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

model = load_model('square_classifier_v2.h5')

for board_img, fname in zip(board_imgs, board_filenames):
    
    squares, names = extract_squares(board_img)
    X = np.array(squares)
    X = X.reshape(X.shape[0], 64, 64, 1)
    X = X.astype('float32')
    X /= 255

    predictions = model.predict(X)
    predictions = np.argmax(predictions, axis=1)

    for pred, sq, name in zip(predictions, squares, names):
        dir_name = dir_names[pred]
        filename = name + "_" + fname
        cv2.imwrite(out_dir + dir_name + "/" + filename, sq)