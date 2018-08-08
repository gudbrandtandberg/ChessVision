import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.utils import to_categorical
import numpy as np
import quilt
import cv2
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN, SMOTETomek

from square_classifier import build_square_classifier
import cv_globals

import time
from collections import Counter

import argparse

labels = {"b": 0, "k": 1, "n": 2, "p": 3, "q": 4, "r": 5, "B": 6,
          "f": 7, "K": 8, "N": 9, "P": 10, "Q": 11, "R": 12}

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=5,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=5
)
valid_datagen = ImageDataGenerator(
    rescale=1./255,
)


def install_data():
    quilt.install("gudbrandtandberg/chesspieces", force=True)


def get_data(node, N):
    X = np.zeros((N, 64, 64, 1))
    y = np.zeros((N))
    i = 0
    dirs = node._group_keys()
    for dir_name, dir_node in zip(dirs, node):
        label = labels[dir_name]
        for img in dir_node:
            # load greyscale image
            X[i, :, :, 0] = cv2.imread(img(), 0)
            y[i] = label
            i += 1
    return X, y


def sample_data(X, y, sample_type):
    print('Sampling data...\nOriginal dataset shape: {}'.format(Counter(y)))

    if sample == "over":
        sampler = SMOTE(random_state=42)
    elif sample == "under":
        sampler = NearMiss(random_state=42)
    else:
        raise Exception("Sampler must be either 'under' under or 'over'")

    X = X.reshape((X.shape[0], 64*64))
    X_sampled, y_sampled = sampler.fit_sample(X, y)
    X_sampled = X_sampled.reshape((X_sampled.shape[0], 64, 64, 1))

    print('Resampled dataset shape: {}'.format(Counter(y_sampled)))
    return X_sampled, y_sampled


def keras_generator(*, transform=False, sample=None):
    def _keras_generator(node, paths):

        datagen = train_datagen if transform else valid_datagen

        X, y = get_data(node, len(paths))

        if sample is not None:
            X, y = sample_data(X, y, sample)

        datagen.fit(X_sampled)
        y_sampled = to_categorical(y_sampled)
        return datagen.flow(X_sampled, y_sampled)
    return _keras_generator


def get_training_generator(sample=None):
    from quilt.data.gudbrandtandberg import chesspieces as pieces
    return pieces["training"](asa=keras_generator(transform=True, sample=sample))


def get_validation_generator():
    from quilt.data.gudbrandtandberg import chesspieces as pieces
    return pieces["validation"](asa=keras_generator(transform=False, sample=None))


def get_class_weights(generator):
    counter = Counter(generator.classes)
    max_val = float(max(counter.values()))
    class_weights = {class_id: max_val /
                     num_images for class_id, num_images in counter.items()}
    return class_weights


# Build the model
if __name__ == "__main__":

    print("Running training job for square classifier...")

    parser = argparse.ArgumentParser(
        description='Train the ChessVision square extractor')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--sample', type=int, default=0,
                        help='How to sample the training data, 0=none, 1=oversample, 2=undersample')
    opt = parser.parse_args()

    # install_data()
    model = build_square_classifier()
    train_generator = get_training_generator(opt.sample)
    valid_generator = get_validation_generator()

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
                                 save_weights_only=True)]

    start = time.time()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=len(train_generator),
                        epochs=opt.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        validation_steps=len(valid_generator))

    duration = time.time() - start
    print("Training the square classifier took {} minutes and {} seconds".format(
        int(np.floor(duration / 60)), int(np.round(duration % 60))))
