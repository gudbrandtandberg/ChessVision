import sys
import argparse
import cv2
from util import listdir_nohidden, parse_arguments
import uuid
import os
import cv_globals

#python chessvision/data_processing/resize_images.py -d data/new_raw -o data/new_resized

_, indir, outdir = parse_arguments()

for f in listdir_nohidden(indir):
    src = os.path.join(indir, f)
    image = cv2.imread(src)
    resized = cv2.resize(image, cv_globals.INPUT_SIZE, interpolation= cv2.INTER_AREA)    
    dst = os.path.join(outdir, f)
    cv2.imwrite(dst, resized)
    print("resized image {}".format(src))