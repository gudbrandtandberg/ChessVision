import cv2
import numpy as np
from util import listdir_nohidden
import matplotlib.pyplot as plt
import pickle
import csv
import time

if __name__ == "__main__":

    # first store the already labelled examples
    already_labelled = []
    with open("../data/labels.csv", "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            already_labelled.append(row[0])

    # next start labelling new examples
    with open('../data/labels.csv', 'a') as csvfile: 

        w = csv.writer(csvfile)
        filenames = listdir_nohidden("../data/squares/")
        filenames = [f for f in filenames]

        labelled_sess = 0
        for f in filenames:
            if f in already_labelled:
                continue

            img = cv2.imread("../data/squares/" + f)
            
            plt.imshow(img, cmap="gray")
            plt.axis("off")
            plt.show()
        
            label = input("Enter label: ")

            if label == "abort":
                break
            labelled_sess += 1

            print([f, label])
            labelled = len(already_labelled) + labelled_sess
            print("Labelled {} examples, {} to go..".format(labelled, len(filenames) - labelled))
            w.writerow([f, label])
            csvfile.flush()