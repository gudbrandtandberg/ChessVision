import sys
#sys.path.append('/usr/lib/python2.7/site-packages/')
#sys.path.append('/usr/local/lib/python2.7/site-packages/')
import os
from util import listdir_nohidden

#from board import Board
from extract import extractBoards, ignoreContours, largestContour
#from util import showImage, drawPerspective, drawBoundaries, drawLines, drawPoint, drawContour, randomColor
#from line import Line

import random
import cv2
import numpy as np
import argparse

extract_width = 512
extract_height = 512

def main(argv):

    ## Parse args
    parser = argparse.ArgumentParser(description='A chess OCR application.')
    parser.add_argument('-f', metavar='filename', type=str, nargs='+',
                        help='The file to process.')
    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-o', metavar='outdir', type=str, nargs='+',
                        help='The outdir to process.')

    args = parser.parse_args()

    indir = args.d[0]

    if args.f != None: # if -f is present, assume file to process
        filenames = [args.f[0]]
    else:
        filenames = listdir_nohidden(indir)
    
    outdir = args.o[0]

    ## Extract boards
    for filename in filenames:
        print("---- Extracting boards from {} ----".format(filename))

        image = cv2.imread(indir + "/" + filename)
        boards = extractBoards(image, extract_width, extract_height)
        
        if len(boards) == 0:
            print("Did not find any boards..")
        
        i = 0
        for b in boards: 
            cv2.imwrite(outdir + "/x{}_".format(i) + filename, b)
            i += 1
        break


if __name__ == "__main__":
    main(sys.argv)