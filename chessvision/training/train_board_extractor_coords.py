import keras
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import model.u_net as unet
from model.augmentations import randomHueSaturationValue, randomShiftScaleRotate, randomHorizontalFlip
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import data
import numpy as np
import cv2 
import cv_globals
from os.path import join

image_dir = join(cv_globals.CVROOT, "data/Segmentation/images/")
labels_dir = join(cv_globals.CVROOT, "data/Segmentation/labels/")

#input_size = 256
#SIZE = (256, 256)
batch_size = 16
epochs = 100

model = unet.get_unet_coords()
model.summary()

ids_train = data.load_image_and_mask_ids()
ids_train_split, ids_valid_split = train_test_split(ids_train, test_size=0.2, random_state=42)

print('Training on {} samples'.format(len(ids_train_split)))
print('Validating on {} samples'.format(len(ids_valid_split)))

def train_generator():
    while True:
        for start in range(0, len(ids_train_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_train_split))
            ids_train_batch = ids_train_split[start:end]
            for id in ids_train_batch:
                img = cv2.imread('{}{}.JPG'.format(image_dir, id))
                #img = cv2.resize(img, (input_size, input_size))
                #mask = cv2.imread('{}{}.JPG'.format(mask_dir, id), cv2.IMREAD_GRAYSCALE)
                #mask = cv2.resize(mask, (input_size, input_size))
                label = np.load(labels_dir + id + ".npy")
                #print("{}: {}".format(id,label.shape))

                img = randomHueSaturationValue(img,
                                               hue_shift_limit=(-50, 50),
                                               sat_shift_limit=(-5, 5),
                                               val_shift_limit=(-15, 15))
                #img, label = randomShiftScaleRotate(img, label,
                                                   shift_limit=(-0.0625, 0.0625),
                                                   scale_limit=(-0.1, 0.1),
                                                   rotate_limit=(-5, 5))
                #img, mask = randomHorizontalFlip(img, mask)
                #mask = np.expand_dims(mask, axis=2)
                x_batch.append(img)
                y_batch.append(label)
                
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch)
            #y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch


def valid_generator():
    while True:
        for start in range(0, len(ids_valid_split), batch_size):
            x_batch = []
            y_batch = []
            end = min(start + batch_size, len(ids_valid_split))
            ids_valid_batch = ids_valid_split[start:end]
            for id in ids_valid_batch:
                img = cv2.imread('{}{}.JPG'.format(image_dir, id))
                #img = cv2.resize(img, (input_size, input_size))
                #mask = cv2.imread('{}{}.JPG'.format(mask_dir, id), cv2.IMREAD_GRAYSCALE)
                #mask = cv2.resize(mask, (input_size, input_size))
                #mask = np.expand_dims(mask, axis=2)
                label = np.load(labels_dir + id + ".npy")
                x_batch.append(img)
                y_batch.append(label)
            x_batch = np.array(x_batch, np.float32) / 255
            y_batch = np.array(y_batch)
            #y_batch = np.array(y_batch, np.float32) / 255
            yield x_batch, y_batch

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
                             filepath='../weights/best_weights_coord.hdf5',
                             save_best_only=True,
                             save_weights_only=True),
             TensorBoard(log_dir='../logs/coord_logs')]

model.fit_generator(generator=train_generator(),
                    steps_per_epoch=np.ceil(float(len(ids_train_split)) / float(batch_size)),
                    epochs=epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=valid_generator(),
                    validation_steps=np.ceil(float(len(ids_valid_split)) / float(batch_size)))