import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
from tensorflow.keras.utils import to_categorical
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

import wandb
from wandb.keras import WandbCallback

labels = {"b": 6, "k": 7, "n": 8, "p": 9, "q": 10, "r": 11, "B": 0,
          "f": 12, "K": 1, "N": 2, "P": 3, "Q": 4, "R": 5}
inverse_labels = ["B", "K", "N", "P", "Q", "R", "b", "k", "n", "p", "q", "r", "f"]

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
        sampler = NearMiss()
    else:
        raise Exception("Sampler must be either 'under' under or 'over'")

    X = X.reshape((X.shape[0], 64*64))
    X_sampled, y_sampled = sampler.fit_resample(X, y)
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
    print(f"Using class weights {class_weights}")
    return class_weights

# Build the model

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print("Running training job for the square classifier...")
        
        sample = None
        batch_size = 16 # config.batch_size
        epochs = 30 #config.epochs or 30
        optimizer = "adam" #config.optimizer or "adam"
        class_weights = None # get_class_weights()# if config.class_weights else None
        dense_1_size = config.dense_1_size
        dense_2_size = config.dense_2_size

        date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        os.mkdir(os.path.join(cv_globals.classifier_weights_dir, date), 0o777)
        weight_filename = cv_globals.square_weights_train.format(date)

        train_generator = get_training_generator(sample, batch_size=batch_size)
        valid_generator = get_validation_generator(batch_size=batch_size)
        model = build_square_classifier(dense_1_size=dense_1_size, dense_2_size=dense_2_size)

        print(model.summary())

        N_train = len(train_generator)
        N_valid = len(valid_generator)

        print("Training on {} samples".format(N_train*batch_size))
        print("Validating on {} samples".format(N_valid*batch_size))

        model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                    optimizer=optimizer,
                    metrics=['accuracy'])

        callbacks = [EarlyStopping(monitor='val_loss',
                                patience=10,
                                verbose=1,
                                min_delta=1e-4),
                    ReduceLROnPlateau(monitor='val_loss',  
                                    factor=0.1,
                                    patience=5,
                                    verbose=1,
                                    min_delta=1e-4),
                    WandbCallback()]

        start = time.time()
        model.fit(x=train_generator,
                  steps_per_epoch=N_train,
                  epochs=epochs,
                  class_weight=class_weights,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=valid_generator,
                  validation_steps=N_valid)

        duration = time.time() - start
        
        print("Training the square classifier took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))


if __name__ == "__main__":
    sweep_config = {
                'method': 'bayes',
                'metric': {'goal': 'minimize', 'name': 'val_accuracy'},
                'parameters': {
                    'batch_size': {"min": 8, "max": 64},
                    # 'epochs': {'value': 30},
                    # 'optimizer': {'values': ['adam']},
                    # 'sample': {'values': [100]},
                    # 'class_weights': {"values": [True, False]}
                    "dense_1_size": {"min": 80, "max": 140},
                    "dense_2_size": {"min": 30, "max": 40}
                }
    }
    sweep_id = wandb.sweep(sweep_config, project="ChessVision")

    wandb.agent(sweep_id, function=train, count=20)