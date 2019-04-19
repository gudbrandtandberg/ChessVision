from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
import cv_globals

input_shape = (64, 64, 1)
num_classes = 13

def load_classifier(weights=None):
    print("Loading square model..")
    if weights:
        model = load_model(weights)
    else:
        model = build_square_classifier()
    print("\rLoading square model.. DONE")
    return model

def build_square_classifier():
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model