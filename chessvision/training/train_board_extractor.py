import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import u_net as unet
from augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import data
import numpy as np
import cv2 
from parameters import *
from util import listdir_nohidden

input_size = 256
SIZE = (256, 256)
batch_size = 16
epochs = 100

def load_image_and_mask_ids():
    filenames = [f[:-4] for f in listdir_nohidden(image_dir)]
    return filenames

ids_train = data.load_image_and_mask_ids()
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.1, random_state=42)
=======
def get_model():
    model = unet.get_unet_256()
    model.load_weights("../weights/best_weights.hdf5")
    return 
>>>>>>> ec822fa3d3b61723cffb2b328ad5951929bcdb09


def train_generator(ids_train_split, batch_size=batch_size):
    
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread('{}{}.JPG'.format(image_dir, id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread('{}{}.JPG'.format(mask_dir, id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                img, mask = randomShiftScaleRotate(img, mask,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-0, 0))
                img, mask = randomHorizontalFlip(img, mask)
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

def valid_generator(ids_valid_split, batch_size=batch_size):
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:
                img = cv2.imread('{}{}.JPG'.format(image_dir, id))
                img = cv2.resize(img, (input_size, input_size))
                mask = cv2.imread('{}{}.JPG'.format(mask_dir, id), cv2.IMREAD_GRAYSCALE)
                mask = cv2.resize(mask, (input_size, input_size))
                mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(mask)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


if __name__ == "__main__":

    ids_train = load_image_and_mask_ids()
    ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

    print('Training on {} samples'.format(len(ids_train_split)))
    print('Validating on {} samples'.format(len(ids_valid_split)))

    model = get_model()

    callbacks = [EarlyStopping(monitor='val_loss',
                            patience=8,
                            verbose=1,
                            min_delta=1e-4),
                ReduceLROnPlateau(monitor='val_loss',
                                factor=0.1,
                                patience=4,
                                verbose=1,
                                min_delta=1e-4),
                ModelCheckpoint(monitor='val_loss',
                                filepath='../weights/new_best_weights.hdf5',
                                save_best_only=True,
                                save_weights_only=True),
                TensorBoard(log_dir='../logs/segmentation_logs/')]

    model.fit_generator(generator=train_generator(ids_train_split),
                        steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                        epochs=epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=valid_generator(ids_valid_split),
                        validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))
