# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import numpy as np
from camera import cameraModule
from pongBot import pongBot

# HARDWARE CONNECTIONS
CAM_PORT = 4 # OpenCV camera capture ID
DXL_PORT = "/dev/ttyUSB0" # U2D2 Serial Port, OS-dependent (Windows: "COM#", Linux: "/dev/ttyUSB#", Mac: idk)

purple = [(150, 50, 20), (180, 255, 255)]
cam = cameraModule(CAM_PORT, purple, 440, 10,
    px_per_meter_x=1450, # px/m
    px_per_meter_y=1450, # px/m
    bot_offset=0.08, # m
    height_scale=0.5 # pixel height multiplier offset
)

# px_per_meter_x=1450, # px/m
# px_per_meter_y=1450, # px/m
# bot_offset=0.08, # m
# height_scale=0.5 # pixel height multiplier offset offset

bot = pongBot(
    motors_port=DXL_PORT,
    link_length=(0.15, 0.125), 
    link_mass=(0.035, 0.035), 
    link_inertia=(0.001, 0.001), 
    motor_mass=(0.08, 0.08), 
    motor_inertia=(0.007, 0.007),
    gear_ratio=(193, 193),
    K_P=np.diag([3.8, 3.8]),
    K_D=np.diag([.11, .11])
)

# disable torque
bot.motors._torque_enable(0)

while True:
    # Needed to read frame, this test doesn't need ball
    cam.find_ball()

    current_joint_angles = bot.q
    print(f"{current_joint_angles=}")
    pos = bot._forward_kinematics(current_joint_angles)

    print(pos)

    if not cam.playback(bot_pos=pos):
        break

    



