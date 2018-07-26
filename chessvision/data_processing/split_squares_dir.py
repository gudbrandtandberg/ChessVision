"""
Split the dataset into train and validation sets


https://stackoverflow.com/questions/42443936/keras-split-train-test-set-when-using-imagedatagenerator
"""

import shutil
import os
import cv_globals
import numpy as np

def split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, testing_data_pct):
    # Recreate testing and training directories
    if testing_data_dir.count('/') > 1:
        #shutil.rmtree(testing_data_dir, ignore_errors=False)
        #os.makedirs(testing_data_dir)
        print("Successfully cleaned directory " + testing_data_dir)
    else:
        print("Refusing to delete testing data directory " + testing_data_dir + " as we prevent you from doing stupid things!")

    if training_data_dir.count('/') > 1:
        #shutil.rmtree(training_data_dir, ignore_errors=False)
        #os.makedirs(training_data_dir)
        print("Successfully cleaned directory " + training_data_dir)
    else:
        print("Refusing to delete training data directory " + training_data_dir + " as we prevent you from doing stupid things!")

    num_training_files = 0
    num_testing_files = 0

    for subdir, dirs, files in os.walk(all_data_dir):
        category_name = os.path.basename(subdir)

        # Don't create a subdirectory for the root directory
        #print(category_name + " vs " + os.path.basename(all_data_dir))
        if category_name == os.path.basename(all_data_dir):
            continue

        training_data_category_dir = training_data_dir + '/' + category_name
        testing_data_category_dir = testing_data_dir + '/' + category_name

        if not os.path.exists(training_data_category_dir):
            os.mkdir(training_data_category_dir)

        if not os.path.exists(testing_data_category_dir):
            os.mkdir(testing_data_category_dir)
            
        for file in files:
            input_file = os.path.join(subdir, file)
                    
            if np.random.rand(1) < testing_data_pct:
                shutil.copy(input_file, testing_data_dir + '/' + category_name + '/' + file)
                num_testing_files += 1
            else:
                shutil.copy(input_file, training_data_dir + '/' + category_name + '/' + file)
                num_training_files += 1

    print("Processed " + str(num_training_files) + " training files.")
    print("Processed " + str(num_testing_files) + " testing files.")

if __name__ == "__main__":
    
    all_data_dir = os.path.join(cv_globals.CVROOT, "data/squares/all")
    training_data_dir = os.path.join(cv_globals.CVROOT, "data/squares/training")
    testing_data_dir = os.path.join(cv_globals.CVROOT, "data/squares/validation")

    split_dataset_into_test_and_train_sets(all_data_dir, training_data_dir, testing_data_dir, 0.2)
