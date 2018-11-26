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

test_data_dir = cv_globals.data_root + "test/"


def sim(a, b):
    return sum([aa == bb for aa, bb in zip(a, b)]) / len(a)


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


def run_tests(data_generator, extractor, classifier, threshold):
    N = 0
    test_accuracy = 0
    times = []
    results = {"raw_imgs": [],
               "board_imgs": [],
               "predictions": [],
               "chessboards": [],
               "squares": []
               }

    for filename, img in data_generator:
        start = time.time()
        truth_file = test_data_dir + "ground_truth/" + filename[:-4] + ".txt"
        with open(truth_file) as truth:
            true_labels = ast.literal_eval(truth.read())
    
        board_img, _,predictions, chessboard, _, squares = chessvision.classify_raw(img, filename, extractor, classifier, threshold=threshold)
        results["board_imgs"].append(board_img)
        results["raw_imgs"].append(img)
        results["predictions"].append(predictions)
        results["chessboards"].append(chessboard)
        results["squares"].append(squares)

        res = vectorize_chessboard(chessboard)
        test_accuracy += sim(res, true_labels)
        N += 1
        stop = time.time()
        times.append(stop-start)

    test_accuracy /= N

    print("Classified {} raw images".format(N))
    results["avg_time"] = sum(times[:-1]) / (N-1)
    results["acc"] = test_accuracy
    #print("Average time per raw classification: {:.2f}s".format(avg_time))

    return results


if __name__ == "__main__":
    print("Computing test accuracy...")

    extractor = load_extractor()
    classifier = load_classifier()

    test_data_gen = get_test_generator()
    acc = compute_test_accuracy(test_data_gen, extractor, classifier)

    print("Test accuracy: {}".format(acc))
