import cv2
import numpy as np
from util import listdir_nohidden
import argparse

def extract_corners(img):    
# Assuming downward-pointing y-axis, corners are numbered as:
#    0 - 1  
#    |   |
#    3 - 4    
    w, h = img.shape
    ww = int(w / 8)
    hh = int(h / 8)
    
    corner1 = img[0:ww,0:hh]
    corner2 = img[0:ww,h-hh:h]
    corner3 = img[w-ww:w,0:hh]
    corner4 = img[w-ww:w,h-hh:h]
    
    return [corner1, corner2, corner3, corner4]

def sum_corners(c):
    w, h = c[0].shape
    max_sum = w*h
    sums = []
    for crnr in c:
        corner_sum = np.sum(crnr)
        sums.append(corner_sum / max_sum)
    return sums

def rotate_board(b):
    corners = extract_corners(b)
    corner_sums = sum_corners(corners)
    ind = np.argmax(corner_sums)

    if ind == 1 or ind == 2:
        w,h = b.shape
        M = cv2.getRotationMatrix2D((h/2,w/2),-90,1)
        rot = cv2.warpAffine(b,M,(h,w))
    return rot

if __name__ == "__main__":
    # Rotate all rotated board-images
    print("Rotating boards...")

    ## Parse args
    parser = argparse.ArgumentParser(description='A chess OCR application.')

    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')

    args = parser.parse_args()

    board_dir = args.d[0]
    

    board_filenames = listdir_nohidden(board_dir)
    filenames = [f for f in board_filenames]
    board_imgs = [cv2.imread(board_dir+f, 0) for f in filenames]

    for f, b in zip(filenames, board_imgs):
        corners = extract_corners(b)
        corner_sums = sum_corners(corners)
        ind = np.argmax(corner_sums)

        if ind == 1 or ind == 2:
            print("Rotating {}".format(f))
            w,h = b.shape
            M = cv2.getRotationMatrix2D((h/2,w/2),-90,1)
            rot = cv2.warpAffine(b,M,(h,w))
            cv2.imwrite(board_dir + f, rot)
