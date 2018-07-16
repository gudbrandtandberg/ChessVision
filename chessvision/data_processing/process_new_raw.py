from rename_images import rename_images
from resize_images import resize_images
from util import parse_arguments

"""Renames the source images and saves resized copies in the output directory"""

# Usage: 
# python chessvision/data_processing/process_new_raw.py -d data/new_raw -o data/board_extraction/new_raw_resized
_, indir, outdir = parse_arguments()
rename_images(indir)
resize_images(indir, outdir)
