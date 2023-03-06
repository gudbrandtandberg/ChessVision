import tensorflow.keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import time
import argparse
import os
from chessvision.model.square_classifier import build_square_classifier
import chessvision.cv_globals as cv_globals
import datetime
from dataset_utils import get_validation_generator_matrix, get_training_generator_matrix, inverse_labels, get_class_weights, install_data


# def predict_and_log_misclassifications(split):
#     if split == "validation":
#         valid_generator, filenames = get_validation_generator(batch_size=args.batch_size, return_filenames=True)
#     elif split == "train":
#         valid_generator, filenames = get_training_generator(sample=None, batch_size=args.batch_size, return_filenames=True)
    
#     ## Predict all validation examples
#     test_table = wandb.Table(columns=["image", "prediction", "true_label", "filename"])
#     num_batches = len(valid_generator)
#     print(f"Creating table of {split} predictions")
#     for batch in range(num_batches):
#         X, y = valid_generator[batch]
#         predictions = model.predict(X)
        
#         predictions = np.argmax(predictions, axis=1)
#         true_labels = np.argmax(y, axis=1)

#         predictions = [inverse_labels[x] for x in predictions]
#         true_labels = [inverse_labels[x] for x in true_labels]

#         for i in range(len(true_labels)):
#             img = X[i]
#             if predictions[i] != true_labels[i]:
#                 test_table.add_data(wandb.Image(img), predictions[i], true_labels[i], filenames[batch*args.batch_size + i])

#     print("Logging table")
#     test_data_at = wandb.Artifact(f"misclassified_{split}_predictions", type="predictions")
#     test_data_at.add(test_table, "predictions")
#     run.log_artifact(test_data_at)

def return_misclassified_examples(X, y, model):
    y_hat = model.predict(X)
    predictions = np.argmax(y_hat, axis=1)
    true_labels = np.argmax(y, axis=1)
    return X[predictions != true_labels], y[predictions != true_labels]

if __name__ == "__main__":

    print("Running training job for the square classifier ensemble...")
    
    date = datetime.datetime.now().strftime("%m-%d-%Y-%H-%M")
    os.mkdir(os.path.join(cv_globals.classifier_1_weights_dir, date), 0o777)
    os.mkdir(os.path.join(cv_globals.classifier_2_weights_dir, date), 0o777)
    os.mkdir(os.path.join(cv_globals.classifier_3_weights_dir, date), 0o777)
    weights_1_filename = cv_globals.square_1_weights_train.format(date)
    weights_2_filename = cv_globals.square_2_weights_train.format(date)
    weights_3_filename = cv_globals.square_3_weights_train.format(date)

    batch_size = 32
    epochs = 10

    model1 = build_square_classifier(dense_1_size=64, dense_2_size=16)
    model2 = build_square_classifier(dense_1_size=64, dense_2_size=16)
    model3 = build_square_classifier(dense_1_size=64, dense_2_size=16)

    X_train, y_train, _ = get_training_generator_matrix(batch_size=batch_size, return_filenames=False)
    X_valid, y_valid, _ = get_validation_generator_matrix()

    N_train = len(X_train)
    N_valid = len(X_valid)

    print("Training model 1 on {} samples".format(N_train))
    print("Validating on {} samples".format(N_valid*batch_size))

    model1.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model2.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    model3.compile(loss=tensorflow.keras.losses.categorical_crossentropy,
                  optimizer=tensorflow.keras.optimizers.Adam(),
                  metrics=['accuracy'])

    callbacks = [ModelCheckpoint(monitor='val_loss', filepath=weights_1_filename, save_best_only=True, save_weights_only=False)]

    start = time.time()
    
    model1.fit(x=X_train, y=y_train,
              epochs=10,
              verbose=1,
              callbacks=[ModelCheckpoint(monitor='val_loss', filepath=weights_1_filename, save_best_only=True, save_weights_only=False)],
              validation_data=(X_valid, y_valid),)
    
    X_train, y_train = return_misclassified_examples(X_train, y_train, model1)
    print("Training model 2 on {} samples".format(len(X_train)))

    model2.fit(x=X_train, y=y_train,
              epochs=15,
              verbose=1,
              callbacks=[ModelCheckpoint(monitor='accuracy', filepath=weights_2_filename, save_best_only=True, save_weights_only=False)],
              validation_data=(X_valid, y_valid),)
    
    X_train, y_train = return_misclassified_examples(X_train, y_train, model2)
    print("Training model 3 on {} samples".format(len(X_train)))
    
    model3.fit(x=X_train, y=y_train,
              epochs=20,
              verbose=1,
              callbacks=[ModelCheckpoint(monitor='accuracy', filepath=weights_3_filename, save_best_only=True, save_weights_only=False)],
              validation_data=(X_valid, y_valid),)
    
    duration = time.time() - start
    print("Training the square classifier ensemble took {} minutes and {} seconds".format(int(np.floor(duration / 60)), int(np.round(duration % 60))))

    # predict_and_log_misclassifications("train")
    # predict_and_log_misclassifications("validation")

    # Run model on test data
    # from test import load_classifier, load_extractor, get_test_generator, run_tests
    # extractor = load_extractor(weights=cv_globals.board_weights)
    # classifier = load_classifier(weights="")
    # test_data_gen = get_test_generator()
    # results = run_tests(test_data_gen, extractor, classifier)
    