import time
import os
import cv2
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras import optimizers
from tensorflow.keras import backend as K
from utils import Sample

from utils import XboxController
from getkeys import key_check
from grabscreen import grab_screen

# from vjoy import vJoy, ultimate_release
os.chdir("E:")
MODEL_NAME = "RocketLeague/test_model_v2.h5"

DIMENSION = 3
WIDTH = 480
HEIGHT = 270

INPUT_SHAPE = (HEIGHT, WIDTH, DIMENSION)
OUT_SHAPE = 10


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


class Actor(object):
    def __init__(self):
        self.model = create_model(keep_prob=1)
        self.model.load_weights(MODEL_NAME)

        # Init controller for manual override
        self.real_controller = XboxController()

    def get_action(self, screen):
        manual_override = self.real_controller.UpDPad == 1

        if not manual_override:
            # Look
            vec = screen
            vec = np.expand_dims(vec, axis=0)
            # Think
            ai_control = self.model.predict(vec, batch_size=1)[0]
        else:
            joystick = self.real_controller.read()
            joystick[1] *= -1

        ### calibration
        output = [
            round((ai_control[0]), 3),  # L / R
            round((ai_control[1]), 3),  # U / D
            round((ai_control[2]), 3),
            round((ai_control[3]), 3),
            int(round(ai_control[4])),  # A
            int(round(ai_control[5])),  # B
            int(round(ai_control[6])),  # X
            int(round(ai_control[7])),  # Y
            int(round(ai_control[8])),  # B
            int(round(ai_control[9])),  # Throttle
        ]

        ### print to console
        if manual_override:
            print("Manual: " + str(output))
        else:
            print("AI: " + str(output))

        return output


def main():
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    actor = Actor()
    end_episode = False
    while not end_episode:
        screen = grab_screen(region=(0, 40, 1920, 1040))
        last_time = time.time()
        # resize to something a bit more acceptable for a CNN
        screen = cv2.resize(screen, (480, 270))
        # run a color convert:
        screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

        # Think & Act
        cv2.imshow("ai view", screen)

        action = actor.get_action(screen)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break


main()
