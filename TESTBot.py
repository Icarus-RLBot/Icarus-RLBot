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


from rlbot.agents.base_agent import BaseAgent, SimpleControllerState
from rlbot.messages.flat.QuickChatSelection import QuickChatSelection
from rlbot.utils.structures.game_data_struct import GameTickPacket

from util.ball_prediction_analysis import find_slice_at_time
from util.boost_pad_tracker import BoostPadTracker
from util.drive import steer_toward_target
from util.sequence import Sequence, ControlStep
from util.vec import Vec3


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


class IcarusActor(object):
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
            int(round(ai_control[9])),  # T
        ]

        ### print to console
        if manual_override:
            print("Manual: " + str(output))
        else:
            print("AI: " + str(output))

        return output


class MyBot(BaseAgent):
    def __init__(self, name, team, index):
        super().__init__(name, team, index)
        self.active_sequence: Sequence = None
        self.boost_pad_tracker = BoostPadTracker()

    def initialize_agent(self):
        # Set up information about the boost pads now that the game is active and the info is available
        self.boost_pad_tracker.initialize_boosts(self.get_field_info())

    def get_output(self, packet: GameTickPacket) -> SimpleControllerState:
        """
        This function will be called by the framework many times per second. This is where you can
        see the motion of the ball, etc. and return controls to drive your car.
        """

        # Keep our boost pad info updated with which pads are currently active
        self.boost_pad_tracker.update_boost_status(packet)

        # This is good to keep at the beginning of get_output. It will allow you to continue
        # any sequences that you may have started during a previous call to get_output.
        if self.active_sequence and not self.active_sequence.done:
            controls = self.active_sequence.tick(packet)
            if controls is not None:
                return controls

        # Gather some information about our car and the ball
        my_car = packet.game_cars[self.index]
        car_location = Vec3(my_car.physics.location)
        car_velocity = Vec3(my_car.physics.velocity)
        ball_location = Vec3(packet.game_ball.physics.location)

        if car_location.dist(ball_location) > 1500:
            # We're far away from the ball, let's try to lead it a little bit
            ball_prediction = (
                self.get_ball_prediction_struct()
            )  # This can predict bounces, etc
            ball_in_future = find_slice_at_time(
                ball_prediction, packet.game_info.seconds_elapsed + 2
            )
            target_location = Vec3(ball_in_future.physics.location)
            self.renderer.draw_line_3d(
                ball_location, target_location, self.renderer.cyan()
            )
        else:
            target_location = ball_location

        # Draw some things to help understand what the bot is thinking
        self.renderer.draw_line_3d(car_location, target_location, self.renderer.white())
        self.renderer.draw_string_3d(
            car_location,
            1,
            1,
            f"Speed: {car_velocity.length():.1f}",
            self.renderer.white(),
        )
        self.renderer.draw_rect_3d(
            target_location, 8, 8, True, self.renderer.cyan(), centered=True
        )

        if 750 < car_velocity.length() < 800:
            # We'll do a front flip if the car is moving at a certain speed.
            return self.begin_front_flip(packet)

        controls = SimpleControllerState()
        controls.steer = steer_toward_target(my_car, target_location)
        controls.throttle = 1.0
        # You can set more controls if you want, like controls.boost.
        actor = IcarusActor()
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

        controls.steer = action[0]
        # controls.throttle = action[9]
        controls.throttle = 1
        return controls
