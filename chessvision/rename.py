import os
from util import listdir_nohidden
import cv_globals
import uuid

i = 0
bases = [cv_globals.data_root + "squares/validation", cv_globals.data_root + "squares/training"]

tally = {}
for base in bases:
    for dirname in listdir_nohidden(base):
        for fname in listdir_nohidden(base + "/" + dirname):
            if len(fname) not in tally:
                tally[len(fname)] = 0 
            else:   
                tally[len(fname)] += 1
            if len(fname) != 18:
                src = base + "/" + dirname + "/" + fname
                uid = str(uuid.uuid1())
                dst = os.path.join(base, dirname, uid + ".JPG")
                #print("saving long file: {}".format(dst))
                os.rename(src, dst)
            

print(tally)

