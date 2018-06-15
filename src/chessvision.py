import cv2
import numpy as np

import board_extractor
import board_classifier

SIZE = (256, 256)

def classify_raw(path):

    filename = path.split("/")[-1]
    print("Classifying board {}".format(filename))

    ###############################   STEP 1    #########################################

    ## Resize image
    img = cv2.imread(path)
    comp_image = cv2.resize(img, SIZE, interpolation=cv2.INTER_LINEAR)
    
    ## Extract board using CNN model and contour approximation
    
    board_img = board_extractor.extract_board(comp_image, img)

    ###############################   STEP 2    #########################################
    
    FEN, predictions = board_classifier.classify_board(board_img)

    print("Classifying board {} ...DONE".format(filename))
    
    return board_img, predictions, FEN