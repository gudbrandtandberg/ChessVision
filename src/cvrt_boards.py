import cv2

from util import listdir_nohidden
board_dir = "../data/Segmentation/boards/"

for f in listdir_nohidden(board_dir):
    img = cv2.imread(board_dir + f)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(board_dir + f, img)