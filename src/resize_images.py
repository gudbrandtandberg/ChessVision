import sys
import argparse
import cv2
from util import listdir_nohidden
import uuid

SIZE = (512, 512)

def main(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', metavar='indir', type=str, nargs='+',
                        help='The dir to process.')
    parser.add_argument('-o', metavar='outdir', type=str, nargs='+',
                        help='The dir to process.')
    args = parser.parse_args()

    indir = args.d[0]
    outdir = args.o[0]

    for f in listdir_nohidden(indir):

        image = cv2.imread(indir + "/" + f)
        dst = cv2.resize(image, (256, 256), interpolation = cv2.INTER_CUBIC)
        unique_filename = str(uuid.uuid4())        
        cv2.imwrite(outdir + unique_filename + ".JPG", dst)



if __name__ == "__main__":
    main(sys.argv)