import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import numpy as np
import cv2 
import cv_globals
from util import listdir_nohidden
from u_net import load_extractor
import time
import argparse
import quilt
import datetime
import os

from quilt.data.gudbrandtandberg import chessboard_segmentation as chessboards

def install_data():
    quilt.install("gudbrandtandberg/chessboard_segmentation")

def get_data(node, split=None):
    
    images = []
    masks  = []
    
    img_nodes   = node["images"]
    mask_nodes  = node["masks"]

    i = 0
    for img, mask in zip(img_nodes, mask_nodes):
       
        _img  = cv2.imread(img())
        _mask = cv2.imread(mask(), cv2.IMREAD_GRAYSCALE)

        images.append(_img)
        masks.append(_mask)

        i += 1

    images = np.array(images)
    masks  = np.expand_dims(np.array(masks), 3)

    return images, masks

def matrix():
    def _matrix(node, paths):
        images, masks = get_data(node)
        return images, masks
    return _matrix

def get_training_generator(images, masks, batch_size=16):
    N = len(images)
    while True:
        for start in range(0, N, batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, N)
            ids_train_batch = list(range(start,end))
            for id in ids_train_batch:
                img = images[id]
                mask = masks[id]
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-50, 50),
                                               val_shift_limit=(-50, 50))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.1, 0.1),
                                                   scale_limit=(-0.2, 0.2),
                                                   rotate_limit=(-5, 5))
                #img, mask = randomHorizontalFlip(img, mask)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def get_validation_generator(images, masks, batch_size=16):
    N = len(images)
    while True:
        for start in range(0, N, batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, N)
            ids_valid_batch = list(range(start, end))
            for id in ids_valid_batch:
                img = images[id]
                mask = masks[id]
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

if __name__ == "__main__":

    print("Running training job for board extractor...")

    parser = argparse.ArgumentParser(
        description='Train the ChessVision board extractor')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')

    parser.add_argument('--install', actin="store_true",
                        help='whether to install the dataset using quilt')
    args = parser.parse_args()

    if args.install:
        install_data()

    images, masks = chessboards(asa=matrix())
    img_train, img_valid, mask_train, mask_valid = train_test_split(images, masks)

    N_train = len(img_train)
    N_valid = len(img_valid)

    training_generator    = get_training_generator(img_train, mask_train, batch_size=args.batch_size)
    validation_generator  = get_validation_generator(img_valid, mask_valid, batch_size=args.batch_size)

    print('Training on {} samples'.format(N_train))
    print('Validating on {} samples'.format(N_valid))

    model = load_extractor()  # or train from scratch?!

    date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    os.mkdir(os.path.join(cv_globals.extractor_weights_dir, date), 0o777)
    weight_filename = cv_globals.board_weights_train.format(date)

    print(model.summary())

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
                                save_weights_only=True)]

    start = time.time()
    model.fit_generator(generator=training_generator,
                        steps_per_epoch=np.ceil(float(N_train / float(args.batch_size))),
                        epochs=args.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=validation_generator,
                        validation_steps=np.ceil(float(N_valid) / float(args.batch_size)))

    duration = time.time() - start
    print("Training the board_extractor took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))