"""
Clean up the weights directory to only keep the best model weights.
Typical path format:
        /path/to/cvroot/weights/modeltype/date/modeltype-{perf}.hdf
where performance is a float between 0 and 1 (1 is good).
"""

import cv_globals
from util import listdir_nohidden
import os 

def find_best(models):
    best_perf = 0.0
    for i in range(len(models)):
        model = models[i]
        perf = float(model.split("-")[1][:-5])
        if perf > best_perf:
            best = i
    return models[best]

models = ["extractor", "classifier"]

for model in models:
    training_runs = listdir_nohidden(cv_globals.weights_dir + model)
    for run in training_runs:
        models = listdir_nohidden(cv_globals.weights_dir + model + "/" + run)
        best = find_best(models)
        for model_file in models:
            if model_file is not best:
                to_delete = cv_globals.weights_dir + model + "/" + run + "/" + model_file
                print("Removing: {}".format(to_delete))
                os.remove(to_delete)