import sys
import argparse
import cv2
from util import listdir_nohidden, parse_arguments
import uuid
import os
import cv_globals

def resize_images(indir, outdir):
    """Resizes new raw images"""
    for f in listdir_nohidden(indir):
        src = os.path.join(indir, f)
        image = cv2.imread(src)
        resized = cv2.resize(image, cv_globals.INPUT_SIZE, interpolation= cv2.INTER_AREA)    
        dst = os.path.join(outdir, f)
        cv2.imwrite(dst, resized)
        print("resized image {}".format(src))
