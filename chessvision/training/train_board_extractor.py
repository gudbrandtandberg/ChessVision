import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import data
import numpy as np
import cv2 
import cv_globals
from util import listdir_nohidden
from u_net import load_extractor
import time
import argparse

def load_image_and_mask_ids():
    filenames = [f[:-4] for f in listdir_nohidden(cv_globals.image_dir)]
    return filenames

ids_train = data.load_image_and_mask_ids()
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)

def train_generator(ids_train_split, batch_size=16):
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread('{}{}.JPG'.format(cv_globals.image_dir, id))
                mask = cv2.imread('{}{}.JPG'.format(cv_globals.mask_dir, id), cv2.IMREAD_GRAYSCALE)
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-5, 50))
                #img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def valid_generator(ids_valid_split, batch_size=16):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:
                img = cv2.imread('{}{}.JPG'.format(cv_globals.image_dir, id))
                mask = cv2.imread('{}{}.JPG'.format(cv_globals.mask_dir, id), cv2.IMREAD_GRAYSCALE)
                mask = np.expand_dims(mask, axis=2)
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

    opt = parser.parse_args()

    ids_train = data.load_image_and_mask_ids()
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    model = load_extractor()  # or train from scratch?!
    print(model.summary())

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
                                filepath=cv_globals.board_weights_train,
                                save_best_only=True,
                                save_weights_only=True)]

    start = time.time()
    model.fit_generator(generator=train_generator(ids_train_split),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(opt.batch_size)),
                        epochs=opt.epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid_split),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(opt.batch_size)))

    duration = time.time() - start
    print("Training the board_extractor took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))