"""
Makes clean board images from the raw images corresponding to hand-labelled resized images
Usage: python data_processing/classify_new_squares.py
"""

import cv2
import numpy as np
from util import listdir_nohidden
from board_extractor import extract_perspective
import os 
import matplotlib.pyplot as plt
import json

input_dir = "../data/new_raw/"
mask_dir = "../data/board_extraction/masks/"
out_dir = "../data/new_boards/"
json_file = "../data/board_extraction/ChessboardSegmentation.json"

in_files = listdir_nohidden(input_dir)

with open(json_file, "r") as file:
    lines = file.readlines()

for line in lines:  #iterates over all files with masks
    x = json.loads(line)
    url = x["content"]
    filename = url.split("_")[-1]

    if filename in in_files:  #there exists a hi-res raw version of this image.

        img = cv2.imread(os.path.join(input_dir, filename))

        scale = float(img.shape[0])

        points = x["annotation"][0]["points"]
        points = [[scale*p[0], scale*p[1]] for p in points[:-1]]
        points = np.array(points, dtype=np.int32)
        
        board = extract_perspective(img, points, 512, 512)

        board = cv2.cvtColor(board, cv2.COLOR_RGB2GRAY)

        cv2.imwrite(os.path.join(out_dir, filename), board)