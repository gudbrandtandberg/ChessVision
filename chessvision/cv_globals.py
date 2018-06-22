import os

CVROOT = os.getenv("CVROOT")

compute_root = CVROOT + "computeroot/"
user_uploads = computeroot + "user_uploads/"

squares_train_dir = CVROOT + "data/squares/training/"
squares_validation_dir = CVROOT + "data/squares/validation/"

image_dir = CVROOT + "/data/board_extraction/images/"
mask_dir = CVROOT + "/data/board_extraction/masks/"

#square_weights = os.path.join(CVROOT, "weights/best_weights_square.hdf5")
square_weights = os.path.join(CVROOT, "weights/best_weights_square_new.hdf5")
board_weights = os.path.join(CVROOT, "weights/best_weights.hdf5")

INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)
