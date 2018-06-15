import cv2
import numpy as np

import board_extractor
import board_classifier

import matplotlib.pyplot as plt

from square_classifier import build_square_classifier
from u_net import get_unet_256

#import tensorflow as tf

def load_classifier():
    print("Loading square model..")
     
    model = build_square_classifier()
    model.load_weights('../weights/best_weights_square.hdf5', by_name=True)
    model._make_predict_function()
    
    print("Loading square model.. DONE")
    return model

def load_extractor():
    print("Loading board extraction model..")
    
    model = get_unet_256()
    model.load_weights('../weights/best_weights.hdf5', by_name=True)
    model._make_predict_function()

    print("Loading board extraction model.. DONE")
    return model

SIZE = (256, 256)

def classify_raw(path):

    filename = path.split("/")[-1]
    print("Classifying board {}".format(filename))

    ###############################   STEP 1    #########################################

    ## Resize image
    img = cv2.imread(path)
    comp_image = cv2.resize(img, SIZE, interpolation=cv2.INTER_LINEAR)
    
    ## Extract board using CNN model and contour approximation
    
    
    board_model = load_extractor()
    board_img = board_extractor.extract_board(comp_image, img, board_model)
    del board_model
    ###############################   STEP 2    #########################################
    
    
    sq_model = load_classifier()
    FEN, predictions = board_classifier.classify_board(board_img, sq_model)
    del sq_model
    print("Classifying board {} ...DONE".format(filename))
    
    return board_img, predictions, FEN

if __name__ == "__main__":

    infile = "../data/raw/IMG_4386.JPG"
    board_img, _, FEN = classify_raw(infile)
    plt.imshow(board_img)
    print(FEN)
    plt.show()

