import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
from square_classifier import build_square_classifier
import cv_globals
from util import listdir_nohidden
import os
import time
from collections import Counter
import quilt
from quilt.nodes import DataNode, GroupNode
import cv2
from keras.utils import to_categorical

def install_data():
    # force to avoid y/n prompt; does not re-download
    quilt.install("gudbrandtandberg/chesspieces", force=True)

labels = {"b": 0, "k": 1, "n": 2, "p": 3, "q": 4, "r": 5, "B": 6,\
          "f": 7, "K": 8, "N": 9, "P": 10, "Q": 11, "R": 12}

train_datagen = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=5,
                    zoom_range=0.05,
                    width_shift_range=0.1,
                    height_shift_range=0.1,
                    shear_range=5
                    )
valid_datagen = ImageDataGenerator(
                rescale=1./255,
                )

def get_data(node, N):
    
    X = np.zeros((N, 64, 64, 1))
    y = np.zeros((N, 1))
    
    i = 0
    dirs = node._group_keys()
    for directory, dir_node in zip(dirs, node):
        label = labels[directory]
        for img in dir_node:
            img = cv2.imread(img(), 0)
            X[i,:,:,0] = img
            y[i] = label
            i += 1
    
    y = to_categorical(y)
    
    return X, y

def keras_generator(*, transform=False):
    def _keras_generator(node, paths):
        datagen = None
        if transform:
            datagen = train_datagen
        else:
            datagen = valid_datagen
        
        X, y = get_data(node, len(paths))
        datagen.fit(X)
        
        return datagen.flow(X, y)
    return _keras_generator

def get_validation_generator():
    from quilt.data.gudbrandtandberg import chesspieces as pieces
    return pieces["validation"](asa=keras_generator(transform=True))

def get_training_generator():
    from quilt.data.gudbrandtandberg import chesspieces as pieces
    return pieces["training"](asa=keras_generator(transform=False))


def count_examples(path):
        sum = 0
        for d in listdir_nohidden(path):
                sum += len(listdir_nohidden(os.path.join(path, d)))
        return sum

def get_class_weights(generator):
        counter = Counter(generator.classes)                          
        max_val = float(max(counter.values()))       
        class_weights = {class_id : max_val/num_images for class_id, num_images in counter.items()}        
        return class_weights
                

# Build the model
if __name__ == "__main__":

	print("Running training job for square classifier...")

	install_data()

	# do this on the commandline
	#batch_size = 32
	#num_classes = 13
	epochs = 100

	model = build_square_classifier()
	train_generator = get_training_generator()
	valid_generator = get_validation_generator()
	#class_weights = get_class_weights(train_generator)

	print(model.summary())

	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer=keras.optimizers.Adadelta(),
		metrics=['accuracy'])

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
				filepath=cv_globals.square_weights_train,
				save_best_only=True,
				save_weights_only=True)]

	start = time.time()
	model.fit_generator(generator=train_generator,
			steps_per_epoch=len(train_generator),
			epochs=epochs,
			verbose=1,
			callbacks=callbacks,
			validation_data=valid_generator,
			validation_steps=len(valid_generator))

	duration = time.time() - start
	print("Training the square classifier took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))
