import cv2
import numpy as np
from util import listdir_nohidden
from board_extractor import extract_perspective
import os 
import json
import cv_globals
from square_classifier import load_classifier
from board_classifier import classify_board
from extract_squares import extract_squares

dir_names = ["B", "K", "N", "P", "Q", "R", "_b", "_k", "_n", "_p", "_q", "_r", "f"]
label_names = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']            

def get_new_boards(indir, json_file):
    """extracts boards from new raw images using coordinates from the json file"""

    boards = {}
    infiles = listdir_nohidden(indir)

    with open(json_file, "r") as file:
        lines = file.readlines()

    for line in lines:  #iterates over all files with masks
        x = json.loads(line)
        url = x["content"]
        filename = url.split("_")[-1]

        if filename in infiles:  #there exists a hi-res raw version of this image.

            img = cv2.imread(os.path.join(indir, filename))

            scale = float(img.shape[0])

            points = x["annotation"][0]["points"]
            points = [[scale*p[0], scale*p[1]] for p in points[:-1]]
            points = np.array(points, dtype=np.int32)
            
            board = extract_perspective(img, points, cv_globals.BOARD_SIZE)

            board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)
            board = cv2.resize(board, cv_globals.BOARD_SIZE)

            boards[filename] = board
    
    return boards

def classify_boards(boards, outdir):
    """Classifies new boards using the current version of the square extraction model"""
    # ugh, what a mess..! 

    make_out_dirs(outdir)

    num_examples = {}

    for label in dir_names:
        num_examples[label] = 0

    model = load_classifier()

    for filename, board_img in boards.items():
        print("Classifying {}".format(filename))

        squares, names = extract_squares(board_img)

        X = squares.astype('float32')
        X /= 255
        
        predictions = model.predict(X)
        predictions = np.argmax(predictions, axis=1)

        for pred, sq_img, sq_name in zip(predictions, squares, names):
            dir_name = dir_names[pred]
            num_examples[dir_name] += 1
            filename = sq_name + "_" + filename
            cv2.imwrite(os.path.join(outdir, dir_name, filename), sq_img)
            
    print(num_examples)

def make_out_dirs(outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    for d in dir_names:
        if not os.path.isdir(os.path.join(outdir, d)):
            os.mkdir(os.path.join(outdir, d))

if __name__ == "__main__":
    # Usage
    # python chessvision/data_processing/classify_new_squares.py

    indir = os.path.join(cv_globals.CVROOT, "data/new_raw/")
    json_file = os.path.join(cv_globals.CVROOT, "data/board_extraction/ChessboardSegmentation.json")

    boards = get_new_boards(indir, json_file)

    print("Found {} new boards".format(len(boards)))

    out_dir = "data/new_squares/"
    classify_boards(boards, out_dir)
