"""
CNN-based board extraction. 
Methods to extract boards from images.
If run as main, attempts to extract boards from all images in in <indir>,
and outputs results in <outdir>

Usage:
python board_extractor_v2.py -d ../data/Segmentation/images/ -o ./data/Segmentation/boards/
"""

import model.u_net as unet
from util import listdir_nohidden
import argparse
import numpy as np
import cv2
import sys

model = unet.get_unet_256()
model.load_weights('../weights/best_weights.hdf5')

SIZE = (256, 256)

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-o', metavar='outdir', type=str, nargs='+',
                        help='The dir to process.')
    args = parser.parse_args()

    image_dir = args.d[0]
    board_dir = args.o[0]

    for f in listdir_nohidden(image_dir):

        print("Extracting board from {}.....".format(f))

        image = cv2.imread(image_dir + f)
        
        comp_image = cv2.resize(image, (256,256), interpolation=cv2.INTER_LINEAR)

        try:
            board = extract_board(comp_image, image)

        except Exception as e:
            print("Board extraction failed due to unexpected error '{}'.. \nMoving on!".format(e))
            continue
        
        cv2.imwrite(board_dir + "/x_" + f, board)
        print("--SUCCESS!!")

def fix_mask(mask):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask[mask > 80] = 255
    mask[mask <= 80] = 0
    return mask

def extract_board(image, orig):
    #predict chessboard-mask:

    image_batch = np.array([image], np.float32) / 255

    predicted_mask_batch = model.predict(image_batch)
    predicted_mask = predicted_mask_batch[0].reshape(SIZE)
    mask = fix_mask(predicted_mask)
    
    #approximate chessboard-mask with a quadrangle
    approx = find_quadrangle(mask)
    
    orig_size = (orig.shape[0], orig.shape[1])
    approx = scale_approx(approx, orig_size) 
    
    #extract board
    board = extract_perspective(orig, approx, 512, 512)

    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = cv2.flip(board, 1)

    return board

def scale_approx(approx, orig_size):
    sf = orig_size[0]/256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


def rotate_quadrangle(approx):
    if approx[0,0,0] < approx[2,0,0]:
        approx = approx[[3, 0, 1, 2],:,:]
    return approx

def find_quadrangle(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnt = contours[0]
    arclen = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.1*arclen, True)

    if len(approx) == 4:
        approx = rotate_quadrangle(approx)

    return approx

def extract_perspective(image, perspective, w, h):
    
    dest = ((0,0), (w, 0), (w,h), (0, h))

    if perspective is None:
        im_w, im_h,_ = image.shape
        perspective = ((0,0), (im_w, 0), (im_w,im_h), (0, im_h))

    perspective = np.array(perspective, np.float32)
    dest = np.array(dest, np.float32)

    coeffs = cv2.getPerspectiveTransform(perspective, dest)
    return cv2.warpPerspective(image, coeffs, (w, h))


if __name__ == "__main__":
    main(sys.argv)