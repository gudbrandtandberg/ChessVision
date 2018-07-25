import cv2
import json
import numpy as np
from util import listdir_nohidden
import cv2
#import matplotlib.pyplot as plt
import os
from os.path import join
import cv_globals


def make_masks(input_dir, output_dir, image_dir, json_file):
    with open(json_file, "r") as file:
        lines = file.readlines()
        
    w, h = cv_globals.INPUT_SIZE

    for line in lines:
        x = json.loads(line)
        url = x["content"]
        filename = url.split("_")[-1]

        if not os.path.isfile(join(output_dir, filename)):
            points = x["annotation"][0]["points"]
            points = [[w*p[0], h*p[1]] for p in points]
            points = np.array(points, dtype=np.int32)
            
            mask = np.zeros((w, h , 1), np.uint8)
            cv2.fillConvexPoly(mask, points, 255)
            print("Making mask for {}".format(filename))
            cv2.imwrite(join(output_dir, filename), mask)
            
            #os.rename(join(input_dir, filename), join(image_dir, filename))

if __name__ == "__main__":

    #python chessvision/data_processing/make_masks.py

    json_file = join(cv_globals.data_root, "board_extraction/coordinates.json")
    input_dir = join(cv_globals.data_root, "new_raw_resized/")
    output_dir = join(cv_globals.data_root, "board_extraction/masks/")
    image_dir = join(cv_globals.data_root, "data/board_extraction/images/")

    make_masks(input_dir, output_dir, image_dir, json_file)