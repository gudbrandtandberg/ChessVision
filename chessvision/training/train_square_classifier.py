from __future__ import print_function
import keras
#from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
#from keras.layers import Dense, Dropout, Flatten
#from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
#import sys
#import chessvision.data
import numpy as np
from square_classifier import build_square_classifier

num_classes = 13
batch_size = 32
epochs = 12

# input image dimensions
input_shape = (64, 64, 1)

# the data, split between train and test sets

train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
        )

#train_datagen.fit()
#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

train_generator = train_datagen.flow_from_directory(
        '../data/squares_gen2',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

# Build the model

model = build_square_classifier()
model.load_weights("../weights/best_weights_square.hdf5")
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               min_delta=1e-4),
             ModelCheckpoint(monitor='loss',
                             filepath='../weights/best_weights_square_gen3.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='../logs/square_logs/')]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=np.ceil(6266./32.),
                    epochs=100,
                    verbose=1,
                    callbacks=callbacks)