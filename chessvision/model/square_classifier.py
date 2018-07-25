from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization
import keras
import cv_globals

input_shape = (64, 64, 1)
num_classes = 13

def load_classifier():
    print("Loading square model..")
    model = build_square_classifier()
    model.load_weights(cv_globals.square_weights)
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

def build_square_classifier_old():
    model = Sequential()
    ´
    # Conv1
    #model.add(Conv2D(16, (3, 3), 
    #                activation="relu", 
    #                input_shape=input_shape))
    #model.add(Dropout(0.2))
    #model.add(BatchNormalization())
    
    # Conv2
    model.add(Conv2D(16, (3, 3),
                    input_shape=input_shape,
                    activation='relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
      
    # Conv3
    model.add(BatchNormalization())
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    
    return model
