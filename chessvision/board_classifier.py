"""
Classify a 512x512 image of a chessboard.
"""
import numpy as np
import chess
from extract_squares import extract_squares
from util import listdir_nohidden, parse_arguments, BoardExtractionError
import cv_globals

def classify_board(board_img, model, flip=False):
    print("Classifying board..")
    
    squares, names = extract_squares(board_img, flip=flip)
    
    predictions = model.predict(squares)
    
    chessboard = classification_logic(predictions, names)
        
    FEN = chessboard.board_fen(promoted=False)
    print("\rClassifying board.. DONE")
    
    return FEN, predictions, squares

def classification_logic(probs, names):
    
    initial_predictions = np.argmax(probs, axis=1)

    label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

    pred_labels = [label_names[p] for p in initial_predictions]

    pred_labels = check_multiple_kings(pred_labels, probs)
    pred_labels = check_bishops(pred_labels, probs, names)
    pred_labels = check_pawns_not_on_first_rank(pred_labels, probs, names)
    
    board = build_board_from_labels(pred_labels, names)
    
    return board

def check_multiple_kings(pred_labels, probs):
    if pred_labels.count("k") > 1:
        print("Predicted more than two black kings!")
        # all but the most likely black king gets switched to the second most likely piece
    if pred_labels.count("K") > 1:
        print("Predicted more than two white kings!")
        # all but the most likely white king gets switched to the second most likely piece
    return pred_labels

def check_bishops(pred_labels, probs, names):
    # check if more than two dark/light bishops
    # check if dark bishop on light square and vice versa
    return pred_labels

def check_pawns_not_on_first_rank(pred_labels, probs, names):
    first_rank = ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"]
    last_rank = ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"]

    for label, name in zip(pred_labels, names):
        if label == "P":
            if name in first_rank:
                print("White pawn on 1st rank!")
        if label == "p":
            if name in last_rank:
                print("Black pawn on 8th rank!")

    return pred_labels

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

if __name__ == "__main__":
    print("No main fn implemented")

    # _, board_dir, svg_dir = parse_arguments()

    # board_filenames = listdir_nohidden(board_dir)
    # board_filenames = [b for b in board_filenames]
    # board_imgs = [cv2.imread(board_dir+f, 0) for f in board_filenames]

    # for board_img, fname in zip(board_imgs, board_filenames):