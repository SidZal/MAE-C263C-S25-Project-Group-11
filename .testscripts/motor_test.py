# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from servosChainClass import servos
from controller import InverseDynamicsController

PORT = '/dev/ttyUSB0'

if __name__ == "__main__":

    # Motor Positions
    # ------------------------------------------------------------------------------
    # Leftmost Configuration = [335, 240]
    # Center Configuration = [300, 220]
    # Rightmost Configuration = [255, 245]
    # ------------------------------------------------------------------------------

    # Gain Matrices
    K_P = np.diag([3.8, 3.8])
    K_D = np.diag([.11, .11])
    
    # Create `DynamixelIO` object to store the serial connection to U2D2
    motors = servos(port=PORT, num_motors=2)

    # Get joint angles
    q_initial = [270, 270]
    q_desired = [240, 330]
    
    # Make controller
    controller = InverseDynamicsController(
        motors=motors,
        K_P=K_P,
        K_D=K_D,
        q_initial_deg=q_initial,
        q_desired_deg=q_desired,
        max_duration_s=3.0
    )

    # Run controller
    controller.start_control_loop()

    # Extract results
    time_stamps = np.asarray(controller.time_stamps)
    joint_positions = np.rad2deg(controller.joint_position_history).T

    # ----------------------------------------------------------------------------------
    # Plot joint positions of the manipulator versus time using the `joint_positions`
    # and `time_stamps` variables, respectively.
    # ----------------------------------------------------------------------------------
    date_str = datetime.now().strftime("%d-%m_%H-%M-%S")
    fig_file_name = f"joint_positions_vs_time_{date_str}.pdf"

    # Create figure and axes
    fig = plt.figure(figsize=(10, 5))
    ax_motor0 = fig.add_subplot(121)
    ax_motor1 = fig.add_subplot(122)

    # Label Plots
    fig.suptitle(f"Motor Angles vs Time")
    ax_motor0.set_title("Motor Joint 0")
    ax_motor1.set_title("Motor Joint 1")
    ax_motor0.set_xlabel("Time [s]")
    ax_motor1.set_xlabel("Time [s]")
    ax_motor0.set_ylabel("Motor Angle [deg]")
    ax_motor1.set_ylabel("Motor Angle [deg]")

    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]), 
        ls="--", 
        color="red", 
        label="Setpoint"
    )
    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]) - 1, ls=":", color="blue"
    )
    ax_motor0.axhline(
        math.degrees(controller.q_desired_rad[0]) + 1, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    ax_motor0.axvline(1.5, ls=":", color="purple")
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]) - 1, 
        ls=":", 
        color="blue", 
        label="Convergence Bound"
    )
    ax_motor1.axhline(
        math.degrees(controller.q_desired_rad[1]) + 1, ls=":", color="blue"
    )
    ax_motor1.axvline(1.5, ls=":", color="purple")

    # Plot motor angle trajectories
    ax_motor0.plot(
        time_stamps,
        joint_positions[0],
        color="black",
        label="Motor Angle Trajectory",
    )
    ax_motor1.plot(
        time_stamps,
        joint_positions[1],
        color="black",
        label="Motor Angle Trajectory",
    )
    ax_motor0.legend()
    ax_motor1.legend()
    # fig.savefig(fig_file_name)
    # ----------------------------------------------------------------------------------
    plt.show()