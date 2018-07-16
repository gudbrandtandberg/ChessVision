import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from square_classifier import build_square_classifier
import cv_globals
from util import listdir_nohidden
import os
import time

def get_train_generator(batch_size=32):
        
        train_datagen = ImageDataGenerator(
                rescale=1./255,
                samplewise_center=False, # TODO: try using this instead?
                samplewise_std_normalization=False,
                rotation_range=5,
                zoom_range=0.05,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=5
                )

        train_generator = train_datagen.flow_from_directory(
                cv_globals.squares_train_dir,
                target_size=cv_globals.PIECE_SIZE,
                color_mode='grayscale',
                batch_size=batch_size,
                class_mode='categorical')
        
        return train_generator


# TODO: get num examples programatically

def get_validation_generator(batch_size=32):
        
        valid_datagen = ImageDataGenerator(
                rescale=1./255,
                samplewise_center=False,
                samplewise_std_normalization=False,
                )

        valid_generator = valid_datagen.flow_from_directory(
            cv_globals.squares_validation_dir,
            target_size=cv_globals.PIECE_SIZE,
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical')
        
        return valid_generator

def count_examples(path):
        #path = "../data/squares/training"
        sum = 0
        for d in listdir_nohidden(path):
                sum += len(listdir_nohidden(os.path.join(path, d)))
        return sum

# Build the model
if __name__ == "__main__":

        num_train = count_examples(cv_globals.squares_train_dir)
        num_valid = count_examples(cv_globals.squares_validation_dir)

        print("Train: {}, valid: {}".format(num_train, num_valid))

        batch_size = 32
        num_classes = 13
        epochs = 100

        # use class_weights!

        model = build_square_classifier()
        print(model.summary())

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
                                filepath=cv_globals.square_weights_train,
                                save_best_only=True,
                                save_weights_only=True),
                TensorBoard(log_dir=cv_globals.CVROOT + '/logs/square_logs/')]

        start = time.time()
        
        model.fit_generator(generator=get_train_generator(batch_size=batch_size),
                        steps_per_epoch=np.ceil(num_train/batch_size),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=get_validation_generator(batch_size=batch_size),
                        validation_steps=np.ceil(num_valid/batch_size))

        print("Training the square classifier took {} seconds".format(time.time() - start))