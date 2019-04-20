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
from sklearn.utils import class_weight
import time
from collections import Counter
import argparse
import sys, os
from quilt.data.gudbrandtandberg import chesspieces as pieces
from square_classifier import build_square_classifier
import cv_globals
import datetime
from sklearn.utils import shuffle

labels = {"b": 6, "k": 7, "n": 8, "p": 9, "q": 10, "r": 11, "B": 0,
          "f": 12, "K": 1, "N": 2, "P": 3, "Q": 4, "R": 5}

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=0,
    zoom_range=0.05,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0
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


def sample_data(X, y, sample):
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


def keras_generator(*, transform=False, sample=None, batch_size=32):
    def _keras_generator(node, paths):

        datagen = train_datagen if transform else valid_datagen

        X, y = get_data(node, len(paths))

        if sample:
            try:
                X, y = shuffle(X, y)
                frac = int(sample)
                N = X.shape[0]
                n = round(N * frac / 100.)
                X = X[:n]
                y = y[:n]
                print("Only using first {} of {} training examples".format(n, N))
            except ValueError:
                X, y = sample_data(X, y, sample)

        datagen.fit(X)
        y = to_categorical(y)
        return datagen.flow(X, y, batch_size=batch_size)
    return _keras_generator

def get_training_generator(sample=None, batch_size=32):
    return pieces["training"](asa=keras_generator(transform=True, sample=sample, batch_size=batch_size))

def get_validation_generator(batch_size=32):
    return pieces["validation"](asa=keras_generator(transform=False, sample=None, batch_size=batch_size))

def labels_only(node, paths):
    _, y = get_data(node, len(paths))
    return y

def get_class_weights():
    print("Computing class weights")
    y = pieces["training"](asa=labels_only)
    class_weights = class_weight.compute_class_weight('balanced', np.unique(y), y)
    return class_weights

# Build the model
if __name__ == "__main__":

    print("Running training job for the square classifier...")
    
    parser = argparse.ArgumentParser(
        description='Train the ChessVision square extractor')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--sample', type=str, default=None,
                        help='how to sample the training data, over=oversample, under=undersample')
    parser.add_argument('--class_weights', action="store_true",
                        help='whether to use class weights during training (no sampling!)')
    parser.add_argument('--install', action="store_true",
                        help='whether to install the dataset using quilt')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to use for training')
    args = parser.parse_args()
    
    date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    os.mkdir(os.path.join(cv_globals.classifier_weights_dir, date), 0o777)
    weight_filename = cv_globals.square_weights_train.format(date)

    model = build_square_classifier()

    if args.install:
        install_data()
    
    class_weights = get_class_weights() if args.class_weights else None

    train_generator = get_training_generator(args.sample, batch_size=args.batch_size)
    valid_generator = get_validation_generator(batch_size=args.batch_size)

    print(model.summary())

    N_train = len(train_generator)
    N_valid = len(valid_generator)

    print("Training on {} samples".format(N_train*args.batch_size))
    print("Validating on {} samples".format(N_valid*args.batch_size))

    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adadelta(),
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   verbose=1,
                                   epsilon=1e-4),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=weight_filename,
                                 save_best_only=True,
                                 save_weights_only=False)]

    start = time.time()
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=N_train,
                        epochs=args.epochs,
                        class_weight=class_weights,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator,
                        validation_steps=N_valid)
    duration = time.time() - start
    print("Training the square classifier took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))