import cv2
from keras.models import load_model
from extract_squares import extract_squares
import numpy as np
from util import listdir_nohidden, parse_arguments, BoardExtractionError
import chess

#from square_classifier import build_square_classifier

def load_classifier():
    print("Loading square model..")
    from square_classifier import build_square_classifier
    model = build_square_classifier()
    model.load_weights('../weights/best_weights_square.hdf5')
    #model._make_predict_function()
    
    print("Loading square model.. DONE")
    return model

def classify_board(board_img):
    
    ## Build the model
    #model = load_classifier()
    model = load_classifier()
    print("Classifying board..")
    squares, names = extract_squares(board_img)
    
    predictions = model.predict(squares)
    del model
    
    chessboard = classification_logic(predictions, names)
        
    FEN = chessboard.board_fen(promoted=False)
    print("Classifying board.. DONE")
    
    return FEN, predictions, squares

def classification_logic(predictions, names):
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
    return board

if __name__ == "__main__":
    print("No main fn implemented")


    # _, board_dir, svg_dir = parse_arguments()

    # board_filenames = listdir_nohidden(board_dir)
    # board_filenames = [b for b in board_filenames]
    # board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

    # for board_img, fname in zip(board_imgs, board_filenames):