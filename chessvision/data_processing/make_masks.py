import cv2
import json
import numpy as np
from util import listdir_nohidden
import cv2
#import matplotlib.pyplot as plt
import os
import cv_globals
#run in chessvision
#python make_masks.py

input_dir = "./data/board_extraction/new_raw_resized/"
output_dir = "./data/board_extraction/masks/"
image_dir = "./data/board_extraction/images/"
json_file = "./data/board_extraction/ChessboardSegmentation.json"

with open(json_file, "r") as file:
    lines = file.readlines()
    
w, h = cv_globals.INPUT_SIZE

for line in lines:
    x = json.loads(line)
    url = x["content"]
    filename = url.split("_")[-1]

    if not os.path.isfile(os.path.join(output_dir, filename)):
        points = x["annotation"][0]["points"]
        points = [[w*p[0], h*p[1]] for p in points]
        points = np.array(points, dtype=np.int32)
        
        mask = np.zeros((w, h , 1), np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        cv2.imwrite(output_dir + filename, mask)

        os.rename(os.path.join(input_dir, filename), os.path.join(image_dir, filename))