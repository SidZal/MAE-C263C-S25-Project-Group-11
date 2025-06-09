import numpy as np

# Project modules
from camera import cameraModule
from pongBot import pongBot


# HARDWARE CONNECTIONS
CAM_PORT = 4 # OpenCV camera capture ID
DXL_PORT = "/dev/ttyUSB0" # U2D2 Serial Port, OS-dependent (Windows: "COM#", Linux: "/dev/ttyUSB#", Mac: idk)

# Camera Module Setup
purple = [(150, 50, 20), (180, 255, 255)] # ball color to look for
cam = cameraModule(
    CAM_PORT, 
    purple, 
    arena_height=400, 
    ball_radius=10,
    endpoint_threshold=200,
    px_per_meter_x=1450,
    px_per_meter_y=1450,
    bot_offset=0.08,
    height_scale=0.5
)

# Pong Bot: takes arm parameters in SI
bot = pongBot(
    motors_port=DXL_PORT,
    link_length=(0.15, 0.125), 
    link_mass=(0.035, 0.035), 
    link_inertia=(0.001, 0.001), 
    motor_mass=(0.08, 0.08), 
    motor_inertia=(0.007, 0.007),
    gear_ratio=(193, 193),
    K_P=np.diag([3.8, 3.8]),
    K_D=np.diag([.11, .11]),
    loop_freq=cam.get_freq(),
    homing_offsets=[-901, -198]
)


while True:
    # Sense: determine desired position, velocity, acceleration
    if cam.find_ball():
        # pass camera prediction to bot
        predicted_endpoint, predicted_dir, predicted_time = cam.predict_path(dt=0.1)
        bot.update_endpoint(predicted_endpoint, predicted_dir, predicted_time)

    # Control: take control step
    pos, pos_d = bot.step()

    # Monitor: display system's actions on camera view
    if not cam.playback(pos, pos_d):
        break

