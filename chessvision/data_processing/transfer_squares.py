import cv_globals
import os
from os.path import join 
from util import listdir_nohidden
import random

indir = join(cv_globals.CVROOT, "computeroot/user_uploads/squares/")
outdir = join(cv_globals.data_root, "squares/")

train_test_split = 0.2

for dirname in listdir_nohidden(indir):
    for filename in listdir_nohidden(join(indir, dirname)):
        split = "validation" if random.random() < train_test_split else "training"
        src = join(indir, dirname, filename)
        dst = join(outdir, split, dirname, filename)
        print("Moving {} to {}".format(src.split("/")[-3:], dst.split("/")[-4:]))
        os.rename(src, dst)    