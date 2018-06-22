import cv2
import numpy as np
from util import listdir_nohidden, parse_arguments

def extract_squares(board, flip=False):
    ranks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    files = ["1", "2", "3", "4", "5", "6", "7", "8"]

    if flip:
        ranks = list(reversed(ranks))
        files = list(reversed(files))
    squares = []
    names = []
    ww, hh = board.shape
    w = int(ww / 8)
    h = int(hh / 8)

    for i in range(8):
        for j in range(8):
            squares.append(board[i*w:(i+1)*w, j*h:(j+1)*h])
            names.append(ranks[j]+files[7-i])
    
    squares = np.array(squares)
    squares = squares.reshape(squares.shape[0], 64, 64, 1)
    
    return squares, names

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