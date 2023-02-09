import cv2

from .cv_globals import INPUT_SIZE
from .util import BoardExtractionError

from .board_extractor import extract_board
from .board_classifier import classify_board

def classify_raw(img, filename, board_model, sq_model, flip=False, threshold=80):

    print("Processing image {}".format(filename))
    
    ###############################   STEP 1    #########################################

    ## Resize image
    comp_image = cv2.resize(img, INPUT_SIZE, interpolation=cv2.INTER_AREA)
    
    ## Extract board using CNN model and contour approximation
    try: 
        board_img, mask = extract_board(comp_image, img, board_model, threshold=threshold)
    except BoardExtractionError as e:
        raise e
    ###############################   STEP 2    #########################################
    
    FEN, predictions, chessboard, squares, names = classify_board(board_img, sq_model, flip=flip)

    #print("Processing image {}.. DONE".format(filename))
    
    return board_img, mask, predictions, chessboard, FEN, squares, names


if __name__ == "__main__":

    infile = "../data/raw/IMG_4386.JPG"
    #board_img, _, FEN, squares = classify_raw(infile)

    #print(FEN)
    

