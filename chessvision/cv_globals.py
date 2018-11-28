import os

CVROOT = os.getenv("CVROOT")

compute_root = os.path.join(CVROOT, "computeroot/")
data_root = os.path.join(CVROOT, "data/")

weights_dir = os.path.join(CVROOT, "weights/")

squares_train_dir = os.path.join(CVROOT, "data/squares/training/")
squares_validation_dir = os.path.join(CVROOT, "data/squares/validation/")

image_dir = os.path.join(CVROOT, "data/board_extraction/images/")
mask_dir = os.path.join(CVROOT, "data/board_extraction/masks/")

classifier_weights_dir = os.path.join(CVROOT, "weights/classifier/")
extractor_weights_dir = os.path.join(CVROOT, "weights/extractor/")

square_weights = os.path.join(CVROOT, "weights/best_classifier.hdf5")
board_weights = os.path.join(CVROOT, "weights/best_extractor.hdf5")

square_weights_train = os.path.join(classifier_weights_dir, "{}/classifier_{{epoch:02d}}-{{val_acc:.4f}}.hdf5")
board_weights_train = os.path.join(extractor_weights_dir, "{}/extractor_{{epoch:02d}}-{{val_dice_coeff:.4f}}.hdf5")

INPUT_SIZE = (256, 256)
BOARD_SIZE = (512, 512)
PIECE_SIZE = (64, 64)

labels = {"b": 6, "k": 7, "n": 8, "p": 9, "q": 10, "r": 11, "B": 0,
          "f": 12, "K": 1, "N": 2, "P": 3, "Q": 4, "R": 5}