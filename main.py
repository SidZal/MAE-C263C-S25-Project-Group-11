from numpy.typing import NDArray
import numpy as np

# Project modules
from camera import cameraModule
from pongBot import pongBot


# HARDWARE CONNECTIONS
CAM_PORT = 4 # OpenCV camera capture ID
DXL_PORT = "/dev/ttyUSB0" # U2D2 Serial Port, OS-dependent (Windows: "COM#", Linux: "/dev/ttyUSB#", Mac: idk)

# SETUP PARAMETERS
METERS_PER_PX_X = 0.1 # m/px
METERS_PER_PX_Y = 0.1 # m/px
BOT_OFFSET = 0.2 # m, distance of bot origin from left-most side of frame


# Pong Bot: takes arm parameters in SI
bot = pongBot(
    motors_port=DXL_PORT,
    link_length=(0.15, 0.125), 
    link_mass=(0.035, 0.035), 
    link_inertia=(0.001, 0.001), 
    motor_mass=(0.08, 0.08), 
    motor_inertia=(0.007, 0.007),
    gear_ratio=(193, 193),
    
)

# Camera Module Setup
purple = [(150, 50, 20), (180, 255, 255)] # ball color to look for
cam = cameraModule(CAM_PORT, purple, arena_height=400, ball_radius=10)


def px_to_mtr(arr: NDArray[np.double]):
    return np.asarray((METERS_PER_PX_X*arr[0], METERS_PER_PX_Y*arr[1]))

def mtr_to_px(arr: NDArray[np.double]):
    return np.asarray((METERS_PER_PX_X*arr[0] - BOT_OFFSET, METERS_PER_PX_Y*arr[1]))


continue_game = True
while continue_game:
    # Sense: determine desired position, velocity, acceleration
    if cam.find_ball(dt=0.1):

        # get endpoint (x,y) in pixels
        predicted_endpoint, predicted_dir, predicted_time = cam.predict_path()

        if predicted_endpoint:            
            # Convert endpoint to m, add origin offset
            predicted_endpoint = px_to_mtr(predicted_endpoint)
            predicted_endpoint[0] += BOT_OFFSET

            predicted_dir = px_to_mtr(predicted_dir)

            bot.update_endpoint(predicted_endpoint, predicted_dir, predicted_time)

    # Control: take control step
    pos, pos_d = bot.step()

    # Monitor: display system's actions on camera view
    if not cam.playback(mtr_to_px(pos), mtr_to_px(pos_d)):
        break

