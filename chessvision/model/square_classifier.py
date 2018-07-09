from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras

input_shape = (64, 64, 1)
num_classes = 13

def build_square_classifier():
    model = Sequential()
    
    model.add(Conv2D(16, (3, 3), 
                    activation="relu", 
                    input_shape=input_shape))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(16, (3, 3),
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
      
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(num_classes, activation='softmax'))
    
    return model