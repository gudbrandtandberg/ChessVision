import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from square_classifier import build_square_classifier
import cv_globals

num_classes = 13
batch_size = 32
epochs = 12

train_datagen = ImageDataGenerator(
        rescale=1./255,
        )

#train_datagen.fit()

train_generator = train_datagen.flow_from_directory(
        '../data/squares/',
        target_size=cv_globals.PIECE_SIZE,
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

# Build the model

model = build_square_classifier()

#from keras.utils import plot_model
#plot_model(model, to_file='model.png')

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
                             filepath='../weights/best_weights_square_gen4.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir=cv_globals.CVROOT + 'logs/square_logs/')]

model.fit_generator(generator=train_generator,
                    steps_per_epoch=np.ceil(6266./batch_size),
                    epochs=100,
                    verbose=1,
                    callbacks=callbacks)