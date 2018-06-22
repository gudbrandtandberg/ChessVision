import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from square_classifier import build_square_classifier
import cv_globals

num_classes = 13
batch_size = 32
epochs = 12

def get_validation_generator():
        valid_datagen = ImageDataGenerator(
                rescale=1./255
                )
        valid_generator = valid_datagen.flow_from_directory(
        '../data/squares/validation/',
        target_size=cv_globals.PIECE_SIZE,
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')
        return valid_generator

def get_train_generator():
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                samplewise_center=False,
                rotation_range=10,
                zoom_range=0.05,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=5
                )

        train_generator = train_datagen.flow_from_directory(
                '../data/squares/training/',
                target_size=cv_globals.PIECE_SIZE,
                color_mode='grayscale',
                batch_size=32,
                class_mode='categorical')
        return train_generator

# Build the model

model = build_square_classifier()

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

callbacks = [EarlyStopping(monitor='val_loss',
                           patience=8,
                           verbose=1,
                           min_delta=1e-4),
             ReduceLROnPlateau(monitor='val_loss',
                               factor=0.1,
                               patience=4,
                               verbose=1,
                               epsilon=1e-4),
             ModelCheckpoint(monitor='val_loss',
                             filepath='../weights/best_weights_square_new.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir=cv_globals.CVROOT + 'logs/square_logs/')]

print(model.summary())

model.fit_generator(generator=get_train_generator(),
                    steps_per_epoch=np.ceil(4975./batch_size),
                    epochs=100,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=get_validation_generator(),
                    validation_steps=np.ceil(1291./batch_size))

# use class_weights!