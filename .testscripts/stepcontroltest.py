# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

import math
import numpy as np
import time
from matplotlib import pyplot as plt
from scipy.interpolate import CubicSpline

from stepcontroller import inverseDynamicsControl
from servosChainClass import servos

motors = servos(port='/dev/ttyUSB0', num_motors=2)

controller = inverseDynamicsControl(
            K_P=np.diag([3.8, 3.8]),
            K_D=np.diag([.11, .11]),
            link_length=(0.15, 0.125),
            link_mass=(0.035, 0.035),
            link_inertia=(0.001, 0.001),
            motor_mass=(0.08, 0.08),
            motor_inertia=(0.007, 0.007),
            gear_ratio=(193, 193)
        )

q_final = np.deg2rad([45, 45])

start_time = time.time()
end_time = 3
times = [0.]
joint_positions = np.asarray(motors.read_position)

while times[-1] < end_time:
    q = motors.read_position()
    qdot = motors.read_velocity()

    spl = CubicSpline(
        x = [times[-1], end_time],
        y = [q, q_final],
        axis = 1,
        bc_type = ((1, qdot), (1, np.zeros(2)))
    )

    u = controller.control_step(
        q=q,
        qdot=qdot,
        q_d=spl(times[-1]),
        qdot_d=spl(times[-1], 1),
        qddot_d=spl(times[-1], 2)
    ) 

    motors.set_pwm(u)
    
    times.append(time.time() - start_time)
    np.append(joint_positions, q)

    time.sleep(1/30)

# Extract results
time_stamps = np.asarray(times)
joint_positions = np.rad2deg(joint_positions).T

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
    math.degrees(q_final[0]), 
    ls="--", 
    color="red", 
    label="Setpoint"
)
ax_motor1.axhline(
    math.degrees(q_final[1]), 
    ls="--", 
    color="red", 
    label="Setpoint"
)
ax_motor0.axhline(
    math.degrees(q_final[0]) - 1, ls=":", color="blue"
)
ax_motor0.axhline(
    math.degrees(q_final[0]) + 1, 
    ls=":", 
    color="blue", 
    label="Convergence Bound"
)
ax_motor0.axvline(1.5, ls=":", color="purple")
ax_motor1.axhline(
    math.degrees(q_final[1]) - 1, 
    ls=":", 
    color="blue", 
    label="Convergence Bound"
)
ax_motor1.axhline(
    math.degrees(q_final[1]) + 1, ls=":", color="blue"
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

# ----------------------------------------------------------------------------------
plt.show()