from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, BatchNormalization

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

def build_square_classifier(dense_1_size=128, dense_2_size=30):
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))
    model.add(Flatten(name="embedding_layer"))
    model.add(Dense(dense_1_size, activation='relu'))
    model.add(Dense(dense_2_size, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    return model