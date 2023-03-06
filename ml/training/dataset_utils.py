import quilt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2
from collections import Counter
from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE
from sklearn.utils import class_weight
from tensorflow.keras.utils import to_categorical
from sklearn.utils import shuffle

pieces = quilt.load("gudbrandtandberg/chesspieces", hash="d28b23f6aa44126b23150d4108c1af7219f33fbbaa2f61b1c9152a9864f1c8dd")

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
    filenames = []
    dirs = node._group_keys()
    for dir_name, dir_node in zip(dirs, node):
        label = labels[dir_name]
        for img in dir_node:
            # load greyscale image
            X[i, :, :, 0] = cv2.imread(img(), 0)
            y[i] = label
            filenames.append(img._meta['_system']['filepath'])
            i += 1
    return X, y, filenames


def sample_data(X, y, sample):
    print('Sampling data...\nOriginal dataset shape: {}'.format(Counter(y)))
    if sample == "over":
        sampler = SMOTE(random_state=42)
    elif sample == "under":
        sampler = NearMiss()
    else:
        raise Exception("Sampler must be either 'under' under or 'over'")

    X = X.reshape((X.shape[0], 64*64))
    X_sampled, y_sampled = sampler.fit_sample(X, y)
    X_sampled = X_sampled.reshape((X_sampled.shape[0], 64, 64, 1))
    print('Resampled dataset shape: {}'.format(Counter(y_sampled)))
    return X_sampled, y_sampled

def matrix(*, transform=False, sample=None, batch_size=32, return_filenames=False):
    def _matrix(node, paths):
        # datagen = train_datagen if transform else valid_datagen
        X, y, filenames = get_data(node, len(paths))
        # datagen.fit(X)
        y = to_categorical(y)
        return X, y, filenames
    return _matrix

def keras_generator(*, transform=False, sample=None, batch_size=32, return_filenames=False):
    def _keras_generator(node, paths):

        datagen = train_datagen if transform else valid_datagen

        X, y, filenames = get_data(node, len(paths))

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
        if return_filenames:
            return datagen.flow(X, y, batch_size=batch_size, shuffle=False), filenames
        else:
            return datagen.flow(X, y, batch_size=batch_size)
    return _keras_generator

def get_training_generator_matrix(sample=None, batch_size=32, return_filenames=False):
    return pieces["training"](asa=matrix(transform=True, sample=sample, batch_size=batch_size, return_filenames=return_filenames))

def get_training_generator(sample=None, batch_size=32, return_filenames=False):
    return pieces["training"](asa=keras_generator(transform=True, sample=sample, batch_size=batch_size, return_filenames=return_filenames))

def get_validation_generator_matrix(batch_size=32, return_filenames=False):
    return pieces["validation"](asa=matrix(transform=False, sample=None, batch_size=batch_size, return_filenames=return_filenames))

def get_validation_generator(batch_size=32, return_filenames=False):
    return pieces["validation"](asa=keras_generator(transform=False, sample=None, batch_size=batch_size, return_filenames=return_filenames))

def labels_only(node, paths):
    _, y, _ = get_data(node, len(paths))
    return y

def get_class_weights():
    print("Computing class weights")
    y = pieces["training"](asa=labels_only)
    class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=np.unique(y), y=y)
    class_weights = {c: w for c, w in enumerate(class_weights)}
    return class_weights