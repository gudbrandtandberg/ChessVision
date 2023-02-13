import os
import logging
import cv2
from tensorflow.keras.models import load_model

from .board_classifier import classify_board
from .board_extractor import extract_board
from .cv_globals import INPUT_SIZE, board_weights, square_weights
from .model.u_net import get_unet_256
from .util import BoardExtractionError

logger = logging.getLogger("chessvision")
 
def load_models():
    logger.debug("Loading models..")
    if not os.path.isfile(square_weights):
        raise Exception("Square model file not found")
    if not os.path.isfile(board_weights):
        raise Exception("Board model file not found")
    board_extractor = get_unet_256()
    board_extractor.load_weights(board_weights)
    square_classifier = load_model(square_weights)
    logger.debug("Models loaded")
    return board_extractor, square_classifier


def classify_raw(img, filename="", board_model=None, sq_model=None, flip=False, threshold=80):

    logger.debug("Processing image {}".format(filename))

    if not board_model or not sq_model:
        board_model, sq_model = load_models()
    
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
