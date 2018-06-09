import cv2
from keras.models import load_model
from extract_squares import extract_squares
import numpy as np
from util import listdir_nohidden
import chess
import chess.svg
import sys
import argparse
from model.square_classifier import build_square_classifier

#usage
#python board_classifier.py -d ../data/Segmentation/boards/ -o ../data/Segmentation/SVG/

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-o', metavar='outdir', type=str, nargs='+',
                        help='The dir to process.')
    args = parser.parse_args()

    board_dir = args.d[0]
    svg_dir = args.o[0]

    board_filenames = listdir_nohidden(board_dir)
    board_filenames = [b for b in board_filenames]
    board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

    model = build_square_classifier()
    model.load_weights('../weights/best_weights_square.hdf5')

    for board_img, fname in zip(board_imgs, board_filenames):
        
        squares, names = extract_squares(board_img)
        squares = np.array(squares)
        squares = squares.reshape(squares.shape[0], 64, 64, 1)
        squares = squares.astype('float32')
        squares /= 255

        predictions = model.predict(squares)
        predictions = np.argmax(predictions, axis=1)

        #label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]
        label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

        board = chess.BaseBoard(board_fen=None)

        for pred, sq in zip(predictions, names):
            if label_names[pred] == "f":
                piece = None
            else:
                piece = chess.Piece.from_symbol(label_names[pred])
            
            square = chess.SQUARE_NAMES.index(sq)
            board.set_piece_at(square, piece, promoted=False)
            
        FEN = board.board_fen(promoted=False)
        print("{}".format(FEN))
        svg_obj = chess.svg.board(board=board, coordinates=False, size=220)

        with open(svg_dir + fname[:-4]+".svg", "w") as f:
            f.write(svg_obj)


if __name__ == "__main__":
    main(sys.argv)