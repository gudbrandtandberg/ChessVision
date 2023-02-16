import argparse
import datetime
import os
import sys
import time
from collections import Counter
import numpy as np
import tensorflow
from dataset_utils import (get_class_weights, get_training_generator,
                           get_validation_generator, install_data,
                           inverse_labels)

from quilt.data.gudbrandtandberg import chesspieces as pieces
from sklearn.utils import class_weight, shuffle
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, TensorBoard)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from wandb.keras import WandbCallback

import chessvision.cv_globals as cv_globals
import wandb
from chessvision.model.square_classifier import build_square_classifier


def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        print("Running training job for the square classifier...")
        
        sample = config.get("sample") or None
        batch_size = config.get("batch_size") or 32
        epochs = config.get("epochs") or 30
        optimizer = config.get("optimizer") or "adam"
        class_weights = get_class_weights() if config.get("class_weights") else None
        dense_1_size = config.get("dense_1_size") or 128
        dense_2_size = config.get("dense_2_size") or 30

        # date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
        # os.mkdir(os.path.join(cv_globals.classifier_weights_dir, date), 0o777)
        # weight_filename = cv_globals.square_weights_train.format(date)

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
                    # 'batch_size': {"min": 8, "max": 64},
                    # 'epochs': {'value': 30},
                    # 'optimizer': {'values': ['adam']},
                    # 'sample': {'values': [100]},
                    'class_weights': {"values": [True, False]}
                    # "dense_1_size": {"min": 80, "max": 140},
                    # "dense_2_size": {"min": 30, "max": 40}
                }
    }
    sweep_id = wandb.sweep(sweep_config, project="ChessVision")

    wandb.agent(sweep_id, function=train, count=4)