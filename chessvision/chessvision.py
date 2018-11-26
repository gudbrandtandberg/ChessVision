import cv2
import numpy as np

import cv_globals
from util import BoardExtractionError

import board_extractor
import board_classifier

def classify_raw(img, filename, board_model, sq_model, flip=False, threshold=80):

    #filename = path.split("/")[-1]
    print("Processing image {}".format(filename))
    
    ###############################   STEP 1    #########################################

    ## Resize image
    #img = cv2.imread(path)
    comp_image = cv2.resize(img, cv_globals.INPUT_SIZE, interpolation=cv2.INTER_AREA)
    
    ## Extract board using CNN model and contour approximation
    try: 
        board_img, mask = board_extractor.extract_board(comp_image, img, board_model, threshold=threshold)
    except BoardExtractionError as e:
        raise e
    ###############################   STEP 2    #########################################
    
    FEN, predictions, chessboard, squares = board_classifier.classify_board(board_img, sq_model, flip=flip)
    #del sq_model
    #print("Processing image {}.. DONE".format(filename))
    
    return board_img, mask, predictions, chessboard, FEN, squares


if __name__ == "__main__":

    infile = "../data/raw/IMG_4386.JPG"
    #board_img, _, FEN, squares = classify_raw(infile)

    #print(FEN)
    

