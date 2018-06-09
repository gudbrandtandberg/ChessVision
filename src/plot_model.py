import model.u_net as unet
from model.square_classifier import build_square_classifier
from keras.utils import plot_model

#model = unet.get_unet_256()
#model.load_weights('../weights/best_weights.hdf5')

#model = build_square_classifier()
#model.load_weights('../weights/best_weights_square.hdf5')


model = unet.get_unet_coords()
model.load_weights('../weights/best_weights_coord.hdf5')
plot_model(model, to_file='../img/model_coord.png')