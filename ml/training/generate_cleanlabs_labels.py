from sklearn.model_selection import cross_val_predict
from chessvision.model.square_classifier import load_classifier
import chessvision.cv_globals as cv_globals
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import tensorflow
from dataset_utils import get_training_generator_matrix
import numpy as np

folds = 5
epochs = 20

X, y, filenames = get_training_generator_matrix()
labels = y.argmax(axis=1)

def get_model():
    model = load_classifier()
    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                optimizer=tensorflow.keras.optimizers.Adam(),
                metrics=['accuracy'])
    return model

classifier = KerasClassifier(get_model)
pred_probs = cross_val_predict(classifier, X, y, fit_params={"epochs": epochs}, method="predict_proba", cv=folds)


np.save("labels", labels)
np.save("pred_probs", pred_probs)
