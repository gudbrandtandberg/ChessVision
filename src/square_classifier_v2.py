from __future__ import print_function
import keras
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import sys
import data

batch_size = 32
num_classes = 13
epochs = 12

# input image dimensions
input_shape = (64, 64, 1)

# the data, split between train and test sets

train_datagen = ImageDataGenerator(
        rescale=1./255
        )

train_generator = train_datagen.flow_from_directory(
        '../data/training_data',
        target_size=(64, 64),
        color_mode='grayscale',
        batch_size=32,
        class_mode='categorical')

# Build the model

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit_generator(
        train_generator,
        steps_per_epoch=2000,
        epochs=12)

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

model.save("square_classifier_v2.h5")