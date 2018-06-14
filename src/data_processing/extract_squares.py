import cv2
from util import listdir_nohidden, parse_arguments
import matplotlib.pyplot as plt
from data_processing.data_util import extract_squares

def extract_squares(board):

if __name__ == "__main__":
    # Extract all squares from all boards
    print("Extracting squares...")
    
    #board_dir = "../data/boards/"
    #square_dir = "../data/squares/"
    
    (_, board_dir, square_dir) = parse_arguments()

    board_filenames = listdir_nohidden(board_dir)

    filenames = [f for f in board_filenames]
    board_imgs = [cv2.imread(board_dir+f, 0) for f in filenames]

    for f, b in zip(filenames, board_imgs):    
        squares, names = extract_squares(b)
        for sq, name in zip(squares, names):
            #print("{}{}".format(name, sq.shape))
            #cv2.imwrite(square_dir + name + f, sq)
            pass

    print("\rExtracting squares...DONE")