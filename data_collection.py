import time
import os
import cv2
import numpy as np

from utils import XboxController
from getkeys import key_check
from grabscreen import grab_screen

SAMPLING_RATE = 100
SAVE_RATE = 500

starting_value = 1

os.chdir("E:")

while True:
    file_name = "RocketLeague/data/training_data-{}.npy".format(starting_value)

    if os.path.isfile(file_name):
        print("File exists, moving along", starting_value)
        starting_value += 1
    else:
        print("File does not exist, starting fresh!", starting_value)
        break


def main(file_name, starting_value):
    # Instantiate the gamepad
    controller = XboxController()

    file_name = file_name
    starting_value = starting_value
    training_data = []
    # Countdown
    for i in list(range(4))[::-1]:
        print(i + 1)
        time.sleep(1)

    last_time = time.time()
    paused = False
    print("STARTING RECORDING!!!!!!!!!")

    while True:
        if not paused:
            screen = grab_screen(region=(0, 40, 1920, 1040))
            last_time = time.time()
            # resize to something a bit more acceptable for a CNN
            screen = cv2.resize(screen, (480, 270))
            # run a color convert:
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)

            controller_output = controller.read()
            training_data.append([screen, controller_output])

            last_time = time.time()

            if len(training_data) % SAMPLING_RATE == 0:
                print(len(training_data))

                # Save a checkpoint
                if len(training_data) == SAVE_RATE:
                    np.save(file_name, training_data)
                    print("SAVED TRAINING DATA")
                    training_data = []
                    starting_value += 1
                    file_name = "RocketLeague/data/training_data-{}.npy".format(
                        starting_value
                    )

        keys = key_check()
        # if 'S' in keys:
        #     np.save(file_name, training_data)
        #     print('SAVED TRAINING DATA')

        if "T" in keys:
            if paused:
                paused = False
                print("unpaused!")
                time.sleep(1)
            else:
                print("Pausing!")
                paused = True
                time.sleep(1)


main(file_name, starting_value)
