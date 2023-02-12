import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import numpy as np
import time
import argparse
import os
from square_classifier import build_square_classifier
import cv_globals
import datetime
from dataset_utils import get_validation_generator, get_training_generator, inverse_labels, get_class_weights, install_data
import wandb
from wandb.keras import WandbCallback

run = wandb.init(project="ChessVision", entity="dnardbug", mode="online") # disabled / online / offline

def predict_and_log_misclassifications(split):
    if split == "validation":
        valid_generator, filenames = get_validation_generator(batch_size=args.batch_size, return_filenames=True)
    elif split == "train":
        valid_generator, filenames = get_training_generator(sample=None, batch_size=args.batch_size, return_filenames=True)
    
    ## Predict all validation examples
    test_table = wandb.Table(columns=["image", "prediction", "true_label", "filename"])
    num_batches = len(valid_generator)
    print(f"Creating table of {split} predictions")
    for batch in range(num_batches):
        X, y = valid_generator[batch]
        predictions = model.predict(X)
        
        predictions = np.argmax(predictions, axis=1)
        true_labels = np.argmax(y, axis=1)

        predictions = [inverse_labels[x] for x in predictions]
        true_labels = [inverse_labels[x] for x in true_labels]

        for i in range(len(true_labels)):
            img = X[i]
            if predictions[i] != true_labels[i]:
                test_table.add_data(wandb.Image(img), predictions[i], true_labels[i], filenames[batch*args.batch_size + i])

    print("Logging table")
    test_data_at = wandb.Artifact(f"misclassified_{split}_predictions", type="predictions")
    test_data_at.add(test_table, "predictions")
    run.log_artifact(test_data_at)

if __name__ == "__main__":

    print("Running training job for the square classifier...")
    
    parser = argparse.ArgumentParser(
        description='Train the ChessVision square extractor')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train for')
    parser.add_argument('--sample', type=str, default=None,
                        help='how to sample the training data, over=oversample, under=undersample')
    parser.add_argument('--class_weights', action="store_true",
                        help='whether to use class weights during training (no sampling!)')
    parser.add_argument('--install', action="store_true",
                        help='whether to install the dataset using quilt')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size to use for training')
    args = parser.parse_args()
    
    date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    os.mkdir(os.path.join(cv_globals.classifier_weights_dir, date), 0o777)
    weight_filename = cv_globals.square_weights_train.format(date)

    model = build_square_classifier()

    if args.install:
        install_data()

    class_weights = get_class_weights() if args.class_weights else None

    train_generator = get_training_generator(args.sample, batch_size=args.batch_size, return_filenames=False)
    valid_generator = get_validation_generator(batch_size=args.batch_size, return_filenames=False)

    print(model.summary())

    N_train = len(train_generator)
    N_valid = len(valid_generator)

    print("Training on {} samples".format(N_train*args.batch_size))
    print("Validating on {} samples".format(N_valid*args.batch_size))

    model.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    callbacks = [EarlyStopping(monitor='val_loss',
                               patience=10,
                               verbose=1,
                               min_delta=1e-4),
                 ReduceLROnPlateau(monitor='val_loss',
                                   factor=0.1,
                                   patience=5,
                                   verbose=1,
                                   min_delta=1e-4),
                 ModelCheckpoint(monitor='val_loss',
                                 filepath=weight_filename,
                                 save_best_only=True,
                                 save_weights_only=False),
                 WandbCallback()]

    start = time.time()
    history = model.fit(x=train_generator,
              steps_per_epoch=N_train,
              epochs=args.epochs,
              class_weight=class_weights,
              verbose=1,
              callbacks=callbacks,
              validation_data=valid_generator,
              validation_steps=N_valid)
    duration = time.time() - start
    print("Training the square classifier took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))

    predict_and_log_misclassifications("train")
    predict_and_log_misclassifications("validation")

    # Run model on test data
    # from test import load_classifier, load_extractor, get_test_generator, run_tests
    # extractor = load_extractor(weights=cv_globals.board_weights)
    # classifier = load_classifier(weights="")
    # test_data_gen = get_test_generator()
    # results = run_tests(test_data_gen, extractor, classifier)
    