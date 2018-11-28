import chessvision
import cv_globals
from u_net import load_extractor
from square_classifier import load_classifier
from util import listdir_nohidden
import cv2
import numpy as np
import os
import ast
import time
import itertools
import matplotlib.pyplot as plt 

test_data_dir = cv_globals.data_root + "test/"
labels = ["f", "P", "p", "R", "r", "N", "n", "B", "b", "Q", "q", "K", "k"]

def plot_confusion_mtx(mtx, labels):
    plt.imshow(mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)

    thresh = mtx.max() / 2.
    for i, j in itertools.product(range(mtx.shape[0]), range(mtx.shape[1])):
        plt.text(j, i, format(mtx[i, j], "d"),
                 horizontalalignment="center",
                 color="white" if mtx[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

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
        mtx[t, p] += 1

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
        board_img, mask, predictions, chessboard, _, squares = chessvision.classify_raw(img, filename, extractor, classifier, threshold=threshold)
        stop = time.time()

        truth_file = test_data_dir + "ground_truth/" + filename[:-4] + ".txt"
        with open(truth_file) as truth:
            true_labels = ast.literal_eval(truth.read())

        times.append(stop-start)
        res = vectorize_chessboard(chessboard)
        test_accuracy += sim(res, true_labels)
        confusion_mtx += confusion_matrix(res, true_labels)

        results["board_imgs"].append(board_img)
        results["raw_imgs"].append(img)
        results["predictions"].append(predictions)
        results["chessboards"].append(chessboard)
        results["squares"].append(squares)
        results["filenames"].append(filename)
        results["masks"].append(mask)
        N += 1

    test_accuracy /= N
    
    results["confusion_matrix"] = confusion_mtx
    results["avg_entropy"] = avg_entropy(results["predictions"])
    results["avg_time"] = sum(times[:-1]) / (N-1)
    results["acc"] = test_accuracy
    print("Classified {} raw images".format(N))

    return results


if __name__ == "__main__":
    print("Computing test accuracy...")

    extractor = load_extractor()
    classifier = load_classifier()

    test_data_gen = get_test_generator()
    results = run_tests(test_data_gen, extractor, classifier)

    print("Test accuracy: {}".format(results["acc"]))