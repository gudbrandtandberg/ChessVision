import os

import aquariumlearning as al
import numpy as np
import tensorflow as tf
from dataset_utils import get_training_generator
from PIL import Image

import chessvision.cv_globals as cv_globals
from chessvision.model.square_classifier import load_classifier

model = load_classifier(weights=cv_globals.square_weights)
embedding_model = tf.keras.Model(inputs=model.input, outputs=model.get_layer("dense_2").output)

label_names  = ['B', 'K', 'N', 'P', 'Q', 'R', 'b', 'k', 'n', 'p', 'q', 'r', 'f']

label_map = {
    "_b": "b",
    "B": "B",
    "_k": "k",
    "K": "K",
    "_q": "q",
    "Q": "Q",
    "_r": "r",
    "R": "R",
    "_n": "n",
    "N": "N",
    "_p": "p",
    "P": "P",
    "f": "f",
}

al_client = al.Client()
API_KEY = "***redacted***"
al_client.set_credentials(api_key=API_KEY)

dataset = al.LabeledDataset()
datagen, filenames = get_training_generator(return_filenames=True)

N = len(filenames)

def upload_dataset():
    for i in range(N):
        _id = filenames[i].split("\\")[-1]
        frame = al.LabeledFrame(frame_id=_id)
        path = filenames[i].replace("\\", "/")
        image_url = f"http://localhost:5000/data/squares/{path}"
        split = path.split("/")
        label = split[1]
        visible_label = label_map[label]
        frame.add_user_metadata("filename", filenames[i])
        frame.add_image(image_url=image_url)
        frame.add_label_2d_classification(label_id=_id, classification=visible_label)
        image = Image.open(os.path.join(cv_globals.CVROOT, "data/squares", filenames[i]))
        image_arr = np.array(image.getdata()).reshape((64, 64, 1)) / 255.
        # prediction = model.predict(np.expand_dims(image_arr, 0)).tolist()
        embedding = embedding_model.predict(np.expand_dims(image_arr, 0))[0].tolist()
        frame.add_frame_embedding(embedding=embedding)
        dataset.add_frame(frame)

    al_client.create_dataset(
        "ChessVision", 
        "pieces_test", 
        dataset=dataset, 
        wait_until_finish=True, 
        preview_first_frame=True
    )

def write_inferences_to_file():
    with open("inferences.txt", "w") as f:
        _id = filenames[i].split("\\")[-1]
        image = Image.open(os.path.join(cv_globals.CVROOT, "data/squares", filenames[i]))
        image_arr = np.array(image.getdata()).reshape((64, 64, 1)) / 255.
        prediction = model.predict(np.expand_dims(image_arr, 0))[0]
        
        conf = float(np.max(prediction))
        pred_index = np.argmax(prediction)
        pred_label = label_names[pred_index]
        f.write(f"{_id},{pred_label},{conf}\n")

def upload_inferences():
    inference_set = al.Inferences()
    with open("inferences.txt", "r") as f:
        lines = f.readlines()
        assert len(lines) == N
        for i, line in enumerate(lines):
            _id = filenames[i].split("\\")[-1]
            file_id, pred_label, conf = line.split(",")
            assert file_id == _id
            conf = float(conf)
            frame = al.InferencesFrame(frame_id=_id)
            frame.add_2d_classification(label_id=f"inf_{_id}", classification=pred_label, confidence=conf)
            inference_set.add_frame(frame)

    al_client.create_inferences(
        "ChessVision", 
        "pieces_test",
        "test_inferences_v2",
        inferences=inference_set, 
        wait_until_finish=True,
    )

if __name__ == "__main__":
    upload_inferences()

    #python -m http.server 5000
