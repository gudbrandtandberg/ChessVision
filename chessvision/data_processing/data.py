from util import listdir_nohidden
import numpy as np
import cv2
import csv
import cv_globals

train_test_split = 0.8

label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]

def load_image_and_mask_ids():
    filenames = [f[:-4] for f in listdir_nohidden(cv_globals.image_dir)]
    return filenames

def load_image_ids():
    filenames = [f[:-4] for f in listdir_nohidden(cv_globals.image_dir)]
    return filenames


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()
