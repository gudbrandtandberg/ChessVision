from square_classifier import build_square_classifier
from board_extractor import load_extractor
from tensorflow.keras.utils import plot_model

#model = load_extractor()
model = build_square_classifier()

plot_model(model, to_file='../img/model_coord.png')