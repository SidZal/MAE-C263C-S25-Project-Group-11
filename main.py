import numpy as np

# Project modules
from camera import cameraModule
from pongBot import pongBot


# HARDWARE CONNECTIONS
CAM_PORT = 4 # OpenCV camera capture ID
DXL_PORT = "/dev/ttyUSB0" # U2D2 Serial Port, OS-dependent (Windows: "COM#", Linux: "/dev/ttyUSB#", Mac: idk)

# Camera Module Setup
purple = [(150, 80, 80), (180, 255, 255)] # ball color to look for
cam = cameraModule(
    CAM_PORT, 
    purple, 
    arena_height=440, 
    ball_radius=10,
    px_per_meter_x=1450,
    px_per_meter_y=1450,
    bot_offset=0.13,
    height_scale=0.5,
    endpoint_threshold=100
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
    K_P=np.diag([1.1, 1.1]),
    K_D=np.diag([.08, .09]),
    loop_freq=cam.get_freq(),
    arena_constraints=cam.get_arena_constraints(),
    controller = 'Inverse Dynamics',
    homing_offsets=[-901, -198]
)

max_vel = 0

while True:
    # Sense: determine desired position, velocity, acceleration
    if cam.find_ball():
        cam.predict_path(dt=0.1)

        # In cam module, generate instructions for bot
        cam_info = cam.generate_bot_goal()

        if cam_info[3] is not None:
            ball_vel = np.sqrt(np.sum(np.asarray(cam_info[3])**2))
            if ball_vel > max_vel:
                max_vel = ball_vel
                print(f"{max_vel=}")
        
        bot.update(cam_info)

    # Control: take control step
    pos, pos_d = bot.step()
    # print(f"{pos=}")
    # print(f"{pos_d=}")

    # Monitor: display system's actions on camera view
    if not cam.playback(pos, pos_d):
        break

