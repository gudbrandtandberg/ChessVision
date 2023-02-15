
import os
import random
import shutil
from io import BytesIO

import cairosvg
import chess
import chess.svg
import cv2
import matplotlib.pyplot as plt
from PIL import Image

import chessvision.cv_globals as cv_globals
from chessvision import classify_raw
from chessvision.model.square_classifier import load_classifier
from chessvision.model.u_net import load_extractor

extractor = load_extractor(weights=cv_globals.board_weights)
classifier = load_classifier(weights=cv_globals.square_weights)

def image_generator(years=["2023"], months=["2"], days=["12"]):

    for root, dirs, files in os.walk(os.path.join(cv_globals.data_root, "bucket")):
        if dirs:
            continue
        day = root.split(os.sep)[-1]
        month = root.split(os.sep)[-2]
        year = root.split(os.sep)[-3]
        if day not in days or month not in months or year not in years:
            print(f"Skipping folder {year}/{month}/{day}")
            continue

        print(f"Scanning folder {year}/{month}/{day}")
        for f in files:
            if not f.endswith("JPG"):
                assert False
            img = cv2.imread(os.path.join(root, f))
            yield img, os.path.join(root, f)


def main():
    threshold = 80
    years = ["2023"]
    months = ["2"]
    days = ["15"]

    datagen = image_generator(years=years, months=months, days=days)

    for img, filename in datagen:

        plt.imshow(img)
        plt.show()
        ok = input("Is this OK?")
        
        if ok == "n":
            print(f"Image {filename} is not OK, discarding..")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "trash"))
            os.remove(filename)
            continue

        if random.uniform(a=0, b=1) < 0.05:
            print(f"Copying {filename} to test data")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "new_test"))
            os.remove(filename)
            continue

        try:
            board_img, mask, predictions, chessboard, _, squares, names = classify_raw(
                img,
                filename,
                extractor,
                classifier,
                threshold=threshold
                )
        except Exception as e:
            print(f"Board extraction failed: {e}")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "new_raw"))
            os.remove(filename)
            continue
        
        svg = chess.svg.board(chessboard)
        img_png = cairosvg.svg2png(svg)
        img = Image.open(BytesIO(img_png))
        
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(board_img)
        plt.subplot(1, 2, 2)
        plt.imshow(img)
        plt.show()
        ok = input("Is the classification correct?")

        if ok != "n":
            print("Image correct, continuing")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "correct"))
            os.remove(filename)
            continue

        plt.figure()
        plt.imshow(board_img)
        plt.show()

        ok = input("Does the board look good?")

        if ok == "n":
            print("Board is not good, copying image to new board images")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "new_raw"))
        
        print("Copying board to new squares images")
        shutil.copy(filename, os.path.join(cv_globals.data_root, "new_squares"))

        os.remove(filename)

if __name__ == "__main__":
    main()