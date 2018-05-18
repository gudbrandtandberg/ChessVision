import cv2
from keras.models import load_model
from extract_squares import extract_squares
import numpy as np
from util import listdir_nohidden
import matplotlib.pyplot as plt
import chess
import chess.svg
#from IPython.display import SVG

board_dir = "../data/Boards/"
out_dir = "../data/results/"

def write_fen(fen_string, fname):
    fname = fname[:-4]
    
    with open(out_dir +fname + "_fen.txt", "w") as f:
        f.write(fen_string)

board_filenames = listdir_nohidden(board_dir)
board_filenames = [b for b in board_filenames]
board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

model = load_model('square_classifier_v1.h5')

for board_img, fname in zip(board_imgs, board_filenames):
    
    squares, names = extract_squares(board_img)
    squares = np.array(squares)
    squares = squares.reshape(squares.shape[0], 64, 64, 1)
    squares = squares.astype('float32')
    squares /= 255

    predictions = model.predict(squares)
    predictions = np.argmax(predictions, axis=1)

    label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]

    board = chess.BaseBoard(board_fen=None)
    

    for pred, sq in zip(predictions, names):
        if label_names[pred] == "f":
            piece = None
        else:
            piece = chess.Piece.from_symbol(label_names[pred])
        
        square = chess.SQUARE_NAMES.index(sq)
        board.set_piece_at(square, piece, promoted=False)
        
    FEN = board.board_fen(promoted=False)
    write_fen(FEN, fname)

# plt.figure()
# plt.imshow(board_img, cmap="gray")
# plt.show()

#SVG(chess.svg.board(board=board))