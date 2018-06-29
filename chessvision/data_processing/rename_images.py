from util import listdir_nohidden, parse_arguments
import uuid
import os
import cv_globals

#python chessvision/data_processing/rename_images.py -d data/new_raw/
_, indir, _ = parse_arguments()

for f in listdir_nohidden(indir):
    src = os.path.join(indir, f)
    unique_filename = str(uuid.uuid4())
    dst = os.path.join(indir, unique_filename+".JPG")
    os.rename(src, dst)
    print("renamed: {} to {}".format(src, dst))
    
    
    
    