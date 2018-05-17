from util import listdir_nohidden
import numpy as np
import cv2
import csv

train_test_split = 0.8

label_names = ["R", "r", "K", "k", "Q", "q", "N", "n", "P", "p", "B", "b", "f"]

def load_data():
    data_dir = "../data/squares/"
    filenames = [fname for fname in listdir_nohidden(data_dir)]

    n = len(filenames)
    w = 64
    h = 64

    labels = {}
    with open("../data/labels.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            labels[row[0]] = row[1]

    img = cv2.imread("../data/squares/" + filenames[0], 0)
    

    X = list(map(lambda x: cv2.imread("../data/squares/" + x, 0), filenames))
    X = np.array(X)
    Y = np.empty((n, 1), dtype=int)
    
    # Fill label vector from unpickled dict    
    for i in range(len(filenames)):
        Y[i] = label_names.index(labels[filenames[i]])

    split = int(n * train_test_split)

    x_train = X[0:split,:,:]
    y_train = Y[0:split]
    x_test = X[split:,:,:]
    y_test = Y[split:]

    return (x_train, y_train), (x_test, y_test)


if __name__ == "__main__":

    (x_train, y_train), (x_test, y_test) = load_data()
