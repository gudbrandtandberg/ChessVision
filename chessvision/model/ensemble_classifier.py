from tensorflow.keras.models import load_model
import numpy as np

class EnsembleSquareClassifier:

    def __init__(self, model_files=[], weights=[]) -> None:
        self.models = []
        for model_file in model_files:
            self.models.append(load_model(model_file))
        
        self.weights = weights
        if not weights:
            self.weights = [1./len(model_files)]*len(model_files)

    def predict(self, X):
        predictions = np.zeros((X.shape[0], 13))
        for model, weight in zip(self.models, self.weights):
            predictions += model.predict(X) * weight
        # Don't really need to normalize..
        return predictions
