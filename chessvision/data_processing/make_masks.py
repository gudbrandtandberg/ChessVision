import cv2
import json
import numpy as np
from util import listdir_nohidden
import matplotlib.pyplot as plt
import os 

input_dir = "../data/board_extraction/resized/"
output_dir = "../data/Segmentation/masks/"
json_file = "../data/Segmentation/ChessboardSegmentation.json"

with open(json_file, "r") as file:
    lines = file.readlines()
    
w = 256
h = 256

for line in lines:
    x = json.loads(line)
    url = x["content"]
    filename = url.split("_")[-1]

    if not os.path.isfile(output_dir + filename):
        points = x["annotation"][0]["points"]
        points = [[w*p[0], h*p[1]] for p in points]
        points = np.array(points, dtype=np.int32)
        
        mask = np.zeros((w, h , 1), np.uint8)
        cv2.fillConvexPoly(mask, points, 255)
        cv2.imwrite(output_dir + filename, mask)    
    