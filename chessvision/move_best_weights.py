

import datetime
import cv_globals
import os

date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
os.mkdir(os.path.join(cv_globals.classifier_weights_dir, date), 777)

print(date)