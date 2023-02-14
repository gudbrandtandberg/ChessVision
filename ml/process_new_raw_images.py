
import os
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

def image_generator():
    # imgs = list(map(lambda x: cv2.imread(test_data_dir + "raw/" + x), img_filenames))

    for root, dirs, files in os.walk(os.path.join(cv_globals.CVROOT, "bucket_")):
        # stripped_root = root.replace(path, "").lstrip("\\")
        print(f"Scanning directory {root}")
        for f in files:
            if not f.endswith("JPG"):
                assert False
            img = cv2.imread(os.path.join(root, f))
            # yield img_filenames[i], test_imgs[i]
            yield img, os.path.join(root, f)


def main():
    threshold = 80
    datagen = image_generator()

    for img, filename in datagen:

        plt.imshow(img)
        plt.show()
        ok = input("Is this OK?")
        
        if ok == "n":
            print(f"Image {filename} is not OK, discarding..")
            shutil.copy(filename, os.path.join(cv_globals.data_root, "trash"))
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
            shutil.copy(filename, os.path.join(cv_globals.data_root, "new_boards"))
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
            shutil.copy(filename, os.path.join(cv_globals.data_root, "new_boards"))
        
        print("Copying board to new squares images")
        shutil.copy(filename, os.path.join(cv_globals.data_root, "new_squares"))

        os.remove(filename)

if __name__ == "__main__":
    main()