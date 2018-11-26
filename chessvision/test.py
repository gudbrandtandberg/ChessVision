import chessvision as cv
import cv_globals
from util import listdir_nohidden
import cv2
from util import listdir_nohidden
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
from u_net import load_extractor
from square_classifier import load_classifier
import chessvision
import ast
import time

labels = ["R", "N", "B", "K", "Q", "P", "r", "n", "b", "p", "k", "q", "f"]

test_data_dir = cv_globals.data_root + "test/"

def entropy(dist):
    return sum([-p * np.log(p) for p in dist if p != 0])

def avg_entropy(predictions):
    predictions = np.array(predictions)
    pred_shape = predictions.shape
    if len(pred_shape) == 3: #(N, 64, 13)
        predictions = np.reshape(predictions, (pred_shape[0]*pred_shape[1], pred_shape[2]))
    return sum([entropy(p) for p in predictions]) / len(predictions)

def sim(a, b):
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)

def confusion_matrix(predicted, truth, N=13):
    
    if type(predicted[0]) == str:
        for i in range(len(predicted)):
            predicted[i] = labels.index(predicted[i])
            truth[i]     = labels.index(truth[i])

    mtx = np.zeros((N, N), dtype=int)

    for p, t in zip(predicted, truth):
        mtx[p, t] += 1

    return mtx

def vectorize_chessboard(board):
    """vectorizes a python-chess board from a1 to h8 along ranks (row-major)"""
    res = ["f"] * 64

    piecemap = board.piece_map()

    for piece in piecemap:
        res[piece] = piecemap[piece].symbol()

    return res


def get_test_generator():

    img_filenames = listdir_nohidden(test_data_dir + "raw/")
    test_imgs = np.array(
        list(map(lambda x: cv2.imread(test_data_dir + "raw/" + x), img_filenames)))

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def run_tests(data_generator, extractor, classifier, threshold=80):
    N = 0
    test_accuracy = 0
    times = []
    results = {"raw_imgs": [],
               "board_imgs": [],
               "predictions": [],
               "chessboards": [],
               "squares": [],
               "filenames": [],
               "masks": []
               }

    confusion_mtx = np.zeros((13, 13), dtype=int)

    for filename, img in data_generator:
        start = time.time()
        truth_file = test_data_dir + "ground_truth/" + filename[:-4] + ".txt"
        with open(truth_file) as truth:
            true_labels = ast.literal_eval(truth.read())
    
        board_img, mask, predictions, chessboard, _, squares = chessvision.classify_raw(img, filename, extractor, classifier, threshold=threshold)
        results["board_imgs"].append(board_img)
        results["raw_imgs"].append(img)
        results["predictions"].append(predictions)
        results["chessboards"].append(chessboard)
        results["squares"].append(squares)
        results["filenames"].append(filename)
        results["masks"].append(mask)

        N += 1
        stop = time.time()
        times.append(stop-start)
        res = vectorize_chessboard(chessboard)
        test_accuracy += sim(res, true_labels)
        confusion_mtx += confusion_matrix(res, true_labels)

    test_accuracy /= N
    
    results["confusion_matrix"] = confusion_mtx
    results["avg_entropy"] = avg_entropy(results["predictions"])
    results["avg_time"] = sum(times[:-1]) / (N-1)
    results["acc"] = test_accuracy
    print("Classified {} raw images".format(N))

    return results


if __name__ == "__main__":
    print("Computing test accuracy...")

    # predicted = [0, 1, 2, 3, 0, 1]
    # truth     = [0, 1, 2, 2, 0, 0]

    # mtx = confusion_matrix(predicted, truth, N=4)

    # predicted = ["r", "p", "q", "n", "n", "f"]
    # truth     = ["R", "p", "q", "N", "N", "p"]

    # mtx = confusion_matrix(predicted, truth)

    # print(mtx)

    extractor = load_extractor()
    classifier = load_classifier()

    test_data_gen = get_test_generator()
    results = run_tests(test_data_gen, extractor, classifier)

    print("Test accuracy: {}".format(results["acc"]))
    print("Confusion matrix:")
    print(results["confusion_matrix"])