
import os

CVROOT = os.getenv("CVROOT")

image_dir = "/Users/gudbrand/Programming/Chess/ChessVision/data/board_extraction/images/"
mask_dir = "/Users/gudbrand/Programming/Chess/ChessVision/data/board_extraction/masks/"

square_weights = os.path.join(CVROOT, "weights/best_weights_square.hdf5")
board_weights = os.path.join(CVROOT, "weights/best_weights.hdf5")