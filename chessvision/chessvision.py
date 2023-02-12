import logging

import cv2

from .board_classifier import classify_board
from .board_extractor import extract_board
from .cv_globals import INPUT_SIZE
from .util import BoardExtractionError

logger = logging.getLogger("chessvision")

def classify_raw(img, filename, board_model, sq_model, flip=False, threshold=80):

    logger.debug("Processing image {}".format(filename))
    
    ###############################   STEP 1    #########################################

    ## Resize image
    comp_image = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    
    ## Extract board using CNN model and contour approximation
    logger.debug("Extracting board from image")
    try: 
        board_img, mask = extract_board(comp_image, img, board_model, threshold=threshold)
    except BoardExtractionError as e:
        raise e
    ###############################   STEP 2    #########################################
    
    logger.debug("Classifying squares")
    FEN, predictions, chessboard, squares, names = classify_board(board_img, sq_model, flip=flip)

    logger.debug("Processing image {}.. DONE".format(filename))
    
    return board_img, mask, predictions, chessboard, FEN, squares, names
