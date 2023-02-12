import itertools
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

import cv_globals as cv_globals
from chessvision import classify_raw
from model.square_classifier import load_classifier
from model.u_net import load_extractor
from util import listdir_nohidden

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

label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

def top_k_sim(predictions, truth, k, names):
    """
    predictions: (64, 13) probability distributions
    truth      : (64, 1)  true labels
    k          : is true label in top k predictions

    returns    : fraction of the 64 squares are k-correctly classified
    """
    
    label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']
    sorted_predictions = np.argsort(predictions, axis=1)
    
    top_k = sorted_predictions[:, -k:]
    
    top_k_predictions = np.array([["" for _ in range(k)] for _ in range(64)])
    
    hits = 0

    for i in range(64):        
        for j in range(k):
            top_k_predictions[i, j] = label_names[top_k[i, j]]

    i = 0
    
    for rank in range(1, 9):
        for file in ["a", "b", "c", "d", "e", "f", "g", "h"]:
            square = file + str(rank)
            square_ind = names.index(square)  
            if truth[i] in top_k_predictions[square_ind]:
                hits += 1
            i += 1

    return hits / 64

def confusion_matrix(predicted, truth, N=13):
    predicted = list(predicted)
    truth = list(truth)
    if type(predicted[0]) == str:
        for i in range(len(predicted)):
            predicted[i] = labels.index(predicted[i])
            truth[i]     = labels.index(truth[i])

    mtx = np.zeros((N, N), dtype=int)

    for p, t in zip(predicted, truth):
        mtx[t, p] += 1

    return mtx

def get_hits(mtx):
    N = mtx.shape[0]
    ondiag = 0
    offdiag = 0
    for i in range(N):
        for j in range(N):
            if i == j:
                ondiag += mtx[i,j]
            else:
                offdiag += mtx[i,j]
    return ondiag, offdiag

def vectorize_chessboard1(board):
    """vectorizes a python-chess board from a1 to h8 along ranks (row-major)"""
    res = ["f"] * 64

    piecemap = board.piece_map()

    for piece in piecemap:
        res[piece] = piecemap[piece].symbol()

    return res

def vectorize_chessboard(board):
    
    res = list("f" * 64)
    
    piecemap = board.piece_map()
    
    for i in range(64):
        
        piece = piecemap.get(i)
        if piece:
            res[i] = piece.symbol()

    return "".join(res)

def get_test_generator():
    img_filenames = listdir_nohidden(test_data_dir + "raw/")
    test_imgs = np.array(
        list(map(lambda x: cv2.imread(test_data_dir + "raw/" + x), img_filenames)))

    for i in range(len(test_imgs)):
        yield img_filenames[i], test_imgs[i]


def run_tests(data_generator, extractor, classifier, threshold=80):
    N = 0
    test_accuracy  = 0
    top_2_accuracy = 0
    top_3_accuracy = 0

    times = []
    results = {"raw_imgs": [],
               "board_imgs": [],
               "predictions": [],
               "chessboards": [],
               "squares": [],
               "filenames": [],
               "masks": [],
               "board_accs": []
               }

    confusion_mtx = np.zeros((13, 13), dtype=int)
    errors = 0

    for filename, img in data_generator:
        start = time.time()
        try: 
            board_img, mask, predictions, chessboard, _, squares, names = classify_raw(img, filename, extractor, classifier, threshold=threshold)
        except:
            errors += 1
            continue

        stop = time.time()

        truth_file = test_data_dir + "ground_truth/" + filename[:-4] + ".txt"
        with open(truth_file) as truth:
            true_labels = truth.read()

        top_2_accuracy += top_k_sim(predictions, true_labels, 2, names)
        top_3_accuracy += top_k_sim(predictions, true_labels, 3, names)

        times.append(stop-start)
        res = vectorize_chessboard(chessboard)
        
        this_board_acc = sim(res, true_labels)
        test_accuracy += this_board_acc

        confusion_mtx += confusion_matrix(res, true_labels)

        results["board_accs"].append(this_board_acc)
        results["board_imgs"].append(board_img)
        results["raw_imgs"].append(img)
        results["predictions"].append(predictions)
        results["chessboards"].append(chessboard)
        results["squares"].append(squares)
        results["filenames"].append(filename)
        results["masks"].append(mask)
        N += 1

    test_accuracy  /= N
    top_2_accuracy /= N
    top_3_accuracy /= N

    results["top_2_accuracy"] = top_2_accuracy
    results["top_3_accuracy"] = top_3_accuracy
    results["confusion_matrix"] = confusion_mtx
    results["avg_entropy"] = avg_entropy(results["predictions"])
    results["avg_time"] = sum(times[:-1]) / (N-1)
    results["acc"] = test_accuracy
    results["errors"] = errors
    results["hits"] = get_hits(confusion_mtx)
    print("Classified {} raw images".format(N))

    return results


if __name__ == "__main__":
    print("Computing test accuracy...")

    extractor = load_extractor(weights=cv_globals.board_weights)
    classifier = load_classifier(weights=cv_globals.square_weights)

    test_data_gen = get_test_generator()
    results = run_tests(test_data_gen, extractor, classifier)

    print("Test accuracy: {}".format(results["acc"]))
    # print("Top-1 accuracy: {}".format(results["top_1_accuracy"]))
    print("Top-2 accuracy: {}".format(results["top_2_accuracy"]))
    print("Top-3 accuracy: {}".format(results["top_3_accuracy"]))