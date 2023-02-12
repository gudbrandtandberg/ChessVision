
from cleanlab.filter import find_label_issues
from cleanlab.dataset import health_summary
from dataset_utils import inverse_labels, get_training_generator
import numpy as np

labels = np.load("labels.npy")
pred_probs = np.load("pred_probs.npy")

ordered_label_issues = find_label_issues(
    labels=labels,
    pred_probs=pred_probs,
    return_indices_ranked_by=None,
    n_jobs=1
)

print(ordered_label_issues)
import matplotlib.pyplot as plt
import os
import cv_globals

train_datagen, filenames = get_training_generator(batch_size=1, return_filenames=True)

for i, filename in enumerate(filenames):
    if not ordered_label_issues[i]:
        continue

    plt.imshow(plt.imread(os.path.join(cv_globals.data_root, "squares", filename)))
    # plt.imshow(plt.imread(os.path.join(cv_globals.data_root, "squares", filenames[ordered_label_issues[i]])))

    plt.show()





summary = health_summary(labels, pred_probs, class_names=inverse_labels)

print(summary)
