"""
Classify a 512x512 image of a chessboard.
"""
import numpy as np
import chess
from .data_processing.extract_squares import extract_squares

label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

def classify_board(board_img, model, flip=False):
    #print("Classifying board..")
    
    squares, names = extract_squares(board_img, flip=flip)
    # squares is (batch, 64, 64, 1)
    predictions = model.predict(squares)
    
    chessboard = classification_logic(predictions, names)
        
    FEN = chessboard.board_fen(promoted=False)
    #print("\rClassifying board.. DONE")
    
    return FEN, predictions, chessboard, squares, names

def classification_logic(probs, names):
    
    initial_predictions = np.argmax(probs, axis=1)

    pred_labels = [label_names[p] for p in initial_predictions]
    
    # pred_labels = check_pawns_not_on_first_rank(pred_labels, probs, names)
    # pred_labels = check_multiple_kings(pred_labels, probs, names)
    # pred_labels = check_bishops(pred_labels, probs, names)
    # pred_labels = check_knights(pred_labels, probs, names)
    
    board = build_board_from_labels(pred_labels, names)
    
    return board

def check_multiple_kings(pred_labels, probs, names):
    argsorted_probs = np.argsort(probs)
    sorted_probs = np.take_along_axis(probs, argsorted_probs, axis=-1)

    for piece_descriptor in ["k", "K"]:
        if pred_labels.count(piece_descriptor) > 1:
            print(f"Predicted more than one occurence of '{piece_descriptor}'")

            king_indices_and_probs = [(i, sorted_probs[i][-1]) for i, v in enumerate(pred_labels) if v == piece_descriptor]
            most_likely_king_index = max(king_indices_and_probs, key=lambda x: x[1])[0]
            for king_index, _ in king_indices_and_probs:
                if king_index == most_likely_king_index:
                    print(f"\tKeeping {piece_descriptor} at {names[king_index]}")
                    continue
                next_guess = argsorted_probs[king_index][-2]
                pred_labels[king_index] = label_names[next_guess]
                print(f"\tSwapping {piece_descriptor} at {names[king_index]} to {label_names[next_guess]}")

    return pred_labels

def check_bishops(pred_labels, probs, names):
    # check if more than two dark/light bishops
    # check if dark bishop on light square and vice versa

    argsorted_probs = np.argsort(probs)
    sorted_probs = np.take_along_axis(probs, argsorted_probs, axis=-1)

    dark_squares  = ["a1", "c1", "e1", "g1", "b2", "d2", "f2", "h2", "a3", "c3", "e3", "g3", "b4", "d4", "f4", "h4", "a5", "c5", "e5", "g5", "b6", "d6", "f6", "h6", "a7", "c7", "e7", "g7", "b8", "d8", "f8", "h8"]
    #light_squares = ["b1", "d1", "f1", "h1", "a2", "c2", "e2", "g2", "b3", "d3", "f3", "h3", "a4", "c4", "e4", "g4", "b5", "d5", "f5", "h5", "a6", "c6", "e6", "g6", "b7", "d7", "f7", "h7", "a8", "c8", "e8", "g8"]

    white_bishops_dark_squares  = []
    white_bishops_light_squares = []
    black_bishops_dark_squares  = []
    black_bishops_light_squares = []

    for label, name in zip(pred_labels, names):
        if label == "B":
            if name in dark_squares:
                white_bishops_dark_squares.append(names.index(name))
            else:
                white_bishops_light_squares.append(names.index(name))
        elif label == "b":
            if name in dark_squares:
                black_bishops_dark_squares.append(names.index(name))
            else:
                black_bishops_light_squares.append(names.index(name))

    for piece_descriptor, indices in [
        ("B", white_bishops_dark_squares),
        ("B", white_bishops_light_squares),
        ("b", black_bishops_dark_squares),
        ("b", black_bishops_light_squares),
    ]:
        if len(indices) > 1:
            print(f"More than one {piece_descriptor} on the same color")
            most_likely = max(indices, key=lambda x: sorted_probs[x][-1])
            for index in indices:
                if index == most_likely:
                    print(f"\tKeeping {piece_descriptor} at {names[index]}")
                    continue
                next_guess = argsorted_probs[index][-2]
                pred_labels[index] = label_names[next_guess]
                print(f"\tSwapping {piece_descriptor} at {names[index]} to {label_names[next_guess]}")

    return pred_labels


def check_knights(pred_labels, probs, names):
    # check if more than two white/black knights

    sorted_probs    = np.sort(probs)

    num_white_knights  = 0
    num_black_knights  = 0

    white_knights  = []
    black_knights  = []

    for label, name in zip(pred_labels, names):
        if label == "N":
            num_white_knights += 1
            white_knights.append(name)
        elif label == "n":
            num_black_knights += 1
            black_knights.append(name)

    if len(white_knights) > 2:
        print("More than two white knights")
        for name in white_knights:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("\tAt {} with prob {:.10f}".format(name, prob))

    if len(black_knights) > 2:
        print("More than two black knights")
        for name in black_knights:
            ind = names.index(name)
            prob = sorted_probs[ind][-1]
            print("\tAt {} with prob {:.10f}".format(name, prob))

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