import os
from util import listdir_nohidden
import cv_globals
import uuid

i = 0
bases = [cv_globals.data_root + "squares/validation", cv_globals.data_root + "squares/training"]

for base in bases:
    for dirname in listdir_nohidden(base):
        for fname in listdir_nohidden(base + "/" + dirname):
            if len(fname) > 150:
                src = base + "/" + dirname + "/" + fname
                uid = str(uuid.uuid1())
                dst = os.path.join(base, dirname, uid + ".JPG")
                #print("saving long file: {}".format(dst))
                os.rename(src, dst)
                
                i += 1

print(i)

