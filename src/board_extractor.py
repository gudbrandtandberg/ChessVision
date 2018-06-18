"""
CNN-based board extraction. 
Methods to extract boards from images.
If run as main, attempts to extract boards from all images in in <indir>,
and outputs results in <outdir>

Usage:
python board_extractor.py -d ../data/images/ -o ./data/boards/
"""

from util import listdir_nohidden, ratio, draw_contour, randomColor, parse_arguments, BoardExtractionError
import numpy as np
import cv2
#from tensorflow import Graph, Session
#import matplotlib.pyplot as plt

SIZE = (256, 256)

def load_extractor():
    print("Loading board extraction model..")
    from u_net import get_unet_256
    model = get_unet_256()
    model.load_weights('../weights/best_weights.hdf5')
    #model._make_predict_function()

    print("Loading board extraction model.. DONE")
    return model

def extract_board(image, orig, model):
    
    #predict chessboard-mask:
    print("Extracting board...")
    image_batch = np.array([image], np.float32) / 255
    
    predicted_mask_batch = model.predict(image_batch)

    predicted_mask = predicted_mask_batch[0].reshape(SIZE)
    mask = fix_mask(predicted_mask)
    
    #approximate chessboard-mask with a quadrangle
    approx = find_quadrangle(mask)

    if approx is None:
        print("Contour approximation failed!")
        raise BoardExtractionError()
    
    orig_size = (orig.shape[0], orig.shape[1])
    approx = scale_approx(approx, orig_size) 
    
    #extract board
    board = extract_perspective(orig, approx, 512, 512)

    board = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
    board = cv2.flip(board, 1)
    print("Extracting board... DONE")
    return board

def fix_mask(mask):
    mask *= 255
    mask = mask.astype(np.uint8)
    mask[mask > 80] = 255
    mask[mask <= 80] = 0
    return mask


def scale_approx(approx, orig_size):
    sf = orig_size[0]/256.0
    scaled = np.array(approx * sf, dtype=np.uint32)
    return scaled


def rotate_quadrangle(approx):
    if approx[0,0,0] < approx[2,0,0]:
        approx = approx[[3, 0, 1, 2],:,:]
    return approx

def find_quadrangle(mask):

    #plt.figure()
    #plt.imshow(mask, cmap="gray")
    #plt.show()

    #(_, mask) = cv2.threshold(mask, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    #_, contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    #plt.figure()
    #plt.imshow(mask, cmap="gray")
    #plt.show()

    _, contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
    
    print("Found {} contour(s)".format(len(contours)))
    contours = ignore_contours(mask, contours)
    print("Filtered to {} contour(s)".format(len(contours)))
    if len(contours) == 0:
        return None
    
    #plt.figure()
    #plt.imshow(mask, cmap="gray")
    #plt.show()
    
    #mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #for i in range(len(contours)):
    #    cv2.drawContours(mask, contours, i, randomColor(), thickness=4)

    #plt.figure()
    #plt.imshow(mask)
    #plt.show()
    
    approx = None

    for i in range(len(contours)):
        cnt = contours[i]
        area = cv2.contourArea(cnt)
        
        arclen = cv2.arcLength(cnt, True)
        app = cv2.approxPolyDP(cnt, 0.1*arclen, True)
        
        if len(app) != 4:
            continue

        approx = rotate_quadrangle(app)
        break

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

def ignore_contours(img,
                   contours,
                   min_ratio_bounding=0.6,
                   min_area_percentage=0.35,
                   max_area_percentage=0.95):

    ret = []
    mask_area = float(img.shape[0]*img.shape[1])

    for i in range(len(contours)):
        ca = cv2.contourArea(contours[i])
        ca /= mask_area
        if ca < min_area_percentage or ca > max_area_percentage:
            continue        
        _, _, w, h = tmp = cv2.boundingRect(contours[i])
        if ratio(h,w) < min_ratio_bounding:
            continue
        ret.append(contours[i])

    return ret

def main():

    _, image_dir, board_dir = parse_arguments()

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

if __name__ == "__main__":
    main()
