"""
Classify a 512x512 image of a chessboard.
"""
import numpy as np
import chess
from .data_processing.extract_squares import extract_squares
from .util import listdir_nohidden, parse_arguments, BoardExtractionError

label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

def classify_board(board_img, model, flip=False):
    #print("Classifying board..")
    
    squares, names = extract_squares(board_img, flip=flip)
    
    predictions = model.predict(squares)
    
    chessboard = classification_logic(predictions, names)
        
    FEN = chessboard.board_fen(promoted=False)
    #print("\rClassifying board.. DONE")
    
    return FEN, predictions, chessboard, squares, names

def classification_logic(probs, names):
    
    initial_predictions = np.argmax(probs, axis=1)

    pred_labels = [label_names[p] for p in initial_predictions]

    pred_labels = check_pawns_not_on_first_rank(pred_labels, probs, names)
    pred_labels = check_multiple_kings(pred_labels, probs)
    pred_labels = check_bishops(pred_labels, probs, names)
    
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

    sorted_probs    = np.sort(probs)
    argsorted_probs = np.argsort(probs)

    dark_squares  = ["a1", "c1", "e1", "g1", "b2", "d2", "f2", "h2", "a3", "c3", "e3", "g3", "b4", "d4", "f4", "h4", "a5", "c5", "e5", "g5", "b6", "d6", "f6", "h6", "a7", "c7", "e7", "g7", "b8", "d8", "f8", "h8"]
    #light_squares = ["b1", "d1", "f1", "h1", "a2", "c2", "e2", "g2", "b3", "d3", "f3", "h3", "a4", "c4", "e4", "g4", "b5", "d5", "f5", "h5", "a6", "c6", "e6", "g6", "b7", "d7", "f7", "h7", "a8", "c8", "e8", "g8"]

    num_white_bishops_dark_squares  = 0
    num_white_bishops_light_squares = 0
    num_black_bishops_dark_squares  = 0
    num_black_bishops_light_squares = 0

    white_bishops_dark_squares  = []
    white_bishops_light_squares = []
    black_bishops_dark_squares  = []
    black_bishops_light_squares = []

    for label, name in zip(pred_labels, names):
        if label == "B":
            if name in dark_squares:
                num_white_bishops_dark_squares += 1
                white_bishops_dark_squares.append(name)
            else:
                num_white_bishops_light_squares += 1
                white_bishops_light_squares.append(name)
        elif label == "b":
            if name in dark_squares:
                num_black_bishops_dark_squares += 1
                black_bishops_dark_squares.append(name)
            else:
                num_black_bishops_light_squares += 1
                black_bishops_light_squares.append(name)

    if num_black_bishops_dark_squares > 1:
        print("More than one black dark-squared bishop")
        for name in black_bishops_dark_squares:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("At {} with prob {:.10f}".format(name, prob))

    if num_black_bishops_light_squares > 1:
        print("More than one black light-squared bishop")
        for name in black_bishops_light_squares:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("At {} with prob {:.10f}".format(name, prob))

    if num_white_bishops_dark_squares > 1:
        print("More than one white dark-squared bishop")
        for name in white_bishops_dark_squares:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("At {} with prob {:.10f}".format(name, prob))
            
    if num_white_bishops_light_squares > 1:
        print("More than one white light-squared bishop")
        for name in white_bishops_light_squares:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("At {} with prob {:.10f}".format(name, prob))

    return pred_labels

def check_pawns_not_on_first_rank(pred_labels, probs, names):
    """
    probs is (64, 13)
    pred_labels is (64, 1) containing argmax of probs
    names is (64, 1) string
    """
    first_rank = ["a1", "b1", "c1", "d1", "e1", "f1", "g1", "h1"]
    last_rank = ["a8", "b8", "c8", "d8", "e8", "f8", "g8", "h8"]

    sorted_probs = np.argsort(probs)
    
    for label, name, i in zip(pred_labels, names, range(64)):
        if name in first_rank or name in last_rank:
            if label == "P" or label == "p":
                new_label = label_names[sorted_probs[i][-2]]
                print("Pawn ({}) on first or last rank. Changing to {}.".format(label, new_label))
                if new_label == "P" or new_label == "P":
                    new_label = label_names[sorted_probs[i][-3]]
                    print("Second best prediction is also pawn, using third ({}).".format(new_label))
                pred_labels[i] = new_label

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