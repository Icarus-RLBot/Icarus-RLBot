import os
import glob
from random import shuffle
from tqdm import tqdm

import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from utils import Sample

os.chdir("E:")

FILE_I_END = 33

EPOCHS = 30

DIMENSION = 3
WIDTH = 480
HEIGHT = 270

INPUT_SHAPE = (HEIGHT, WIDTH, DIMENSION)
MODEL_NAME = "test_model_v2"

OUT_SHAPE = 10  # Vector of len 10 where each element is a value for a control output


def customized_loss(y_true, y_pred, loss="euclidean"):
    # Simply a mean squared error that penalizes large joystick summed values
    if loss == "L2":
        L2_norm_cost = 0.001
        val = (
            K.mean(K.square((y_pred - y_true)), axis=-1)
            + K.sum(K.square(y_pred), axis=-1) / 2 * L2_norm_cost
        )
    # euclidean distance loss
    elif loss == "euclidean":
        val = K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))
    return val


def create_model(keep_prob=0.8):
    model = Sequential()

    # NVIDIA's model
    model.add(
        Conv2D(
            24,
            kernel_size=(5, 5),
            strides=(2, 2),
            activation="relu",
            input_shape=INPUT_SHAPE,
        )
    )
    model.add(Conv2D(36, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(48, kernel_size=(5, 5), strides=(2, 2), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
    model.add(Flatten())
    model.add(Dense(1164, activation="relu"))
    drop_out = 1 - keep_prob
    model.add(Dropout(drop_out))
    model.add(Dense(100, activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Dense(50, activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Dense(10, activation="relu"))
    model.add(Dropout(drop_out))
    model.add(Dense(OUT_SHAPE, activation="softsign"))
    return model


# Handy function to properly sort files
def order_files_by_date(path_to_folder, file_type):
    files = glob.glob("%s*%s" % (path_to_folder, file_type))
    files.sort(key=os.path.getmtime)
    return files


data_files = order_files_by_date("RocketLeague/data/", ".npy")
model = create_model()
model.compile(loss=customized_loss, optimizer=optimizers.Adam())
BATCH_SIZE = 50

for e in range(EPOCHS):
    for i, file in enumerate(data_files):
        try:
            data = np.load(file, allow_pickle=True)

            train = data[:-50]
            test = data[-50:]

            X = np.array([i[0] for i in train]).reshape(-1, WIDTH, HEIGHT, 3)
            Y = [i[1] for i in train]

            test_x = np.array([i[0] for i in test]).reshape(-1, WIDTH, HEIGHT, 3)
            test_y = [i[1] for i in test]

            #             model.fit(X, Y, batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True, validation_split=0.1)
            model.fit(
                {"input": X},
                {"targets": Y},
                n_epoch=1,
                validation_set=({"input": test_x}, {"targets": test_y}),
                snapshot_step=2500,
                show_metric=True,
                run_id=MODEL_NAME,
            )
            # model.fit(X, Y, n_epoch=1, validation_set=(test_x, test_y),
            # snapshot_step=2500, show_metric=True, run_id=MODEL_NAME)
            print(e)

        except Exception as ex:
            print(str(ex))

model.save_weights(f"{MODEL_NAME}.h5")

