#!/anaconda/envs/tensorflow/bin/python

import sys
import cv2
from keras.models import load_model
from util import parse_arguments, draw_contour
from data_util import extract_squares, write_fen
import numpy as np
import chess
from board_extractor import extract_board
import u_net as unet
import square_classifier as sq_clf

print("Initiating main function")

#usage: ./main.py -f ../data/Segmentation/hi_res_raw/filename.jpg -o unused

path, _, _ = parse_arguments()   #something like './uploads/img.jpg'
filename = path.split("/")[-1]

print("Extracting board from {}".format(filename))

img = cv2.imread(path)
comp_image = cv2.resize(img, (256,256), interpolation=cv2.INTER_LINEAR)

#cv2.imwrite("test.jpg", board_img)
model = unet.get_unet_256()
model.load_weights('../weights/best_weights.hdf5')

board_img = extract_board(comp_image, img, model)
del model

squares, names = extract_squares(board_img)

squares = np.array(squares)
squares = squares.reshape(squares.shape[0], 64, 64, 1)
squares = squares.astype('float32')
squares /= 255

model = sq_clf.build_square_classifier()
model.load_weights('../weights/best_weights_square.hdf5')

predictions = model.predict(squares)
predictions = np.argmax(predictions, axis=1)

#label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]
label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

board = chess.BaseBoard()

for pred, sq in zip(predictions, names):
    if label_names[pred] == "f":
        piece = None
    else:
        piece = chess.Piece.from_symbol(label_names[pred])
        
    square = chess.SQUARE_NAMES.index(sq)
    board.set_piece_at(square, piece, promoted=False)
        
FEN = board.board_fen(promoted=False)
write_fen(FEN, "../computeroot/user_uploads/" + filename)

print("\rExtracting board from {} ...DONE".format(filename))
print(FEN)