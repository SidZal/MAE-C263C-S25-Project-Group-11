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
from fixedFrequencyLoopManager import fixedFrequencyLoopManager

motors = servos(port='/dev/ttyUSB0', num_motors=2, homing_offsets=[-901, -198])

controller = inverseDynamicsControl(
            K_P=np.diag([3.2, 3]),
            K_D=np.diag([.07, .09]),
            link_length=(0.15, 0.125),
            link_mass=(0.035, 0.035),
            link_inertia=(0.001, 0.001),
            motor_mass=(0.08, 0.08),
            motor_inertia=(0.007, 0.007),
            gear_ratio=(193, 193)
        )

loop_manager = fixedFrequencyLoopManager(30.0)

# q_final = np.asarray(motors.read_position())
q_final = np.deg2rad([0, 0])

end_time = 3
joint_positions = [motors.read_position()]
control_period = 1/30.0

# spl = CubicSpline(
#     x = [0, end_time + .2],
#     y = np.asarray([joint_positions[0], q_final]).T,
#     axis = 1,
#     bc_type = ((1, np.zeros(2)), (1, np.zeros(2)))
# )

# time.sleep(5)
# print("Now")

start_time = time.time()
times = [0.]


while times[-1] < end_time - 1:
    q = np.asarray(motors.read_position())
    qdot = motors.read_velocity()

    times.append(time.time() - start_time)
    joint_positions.append(q)

    time_to = end_time - times[-1]

    spl = CubicSpline(
        x = [0, time_to/2, time_to],
        y = np.asarray([q, (q+q_final) / 2 ,q_final]).T,
        # x = [times[-1], end_time],
        # y = np.asarray([q, q_final]).T,
        axis = 1,
        bc_type = "clamped"
        # bc_type = ((1, qdot), (1, np.zeros(2)))
    )

    print(f"{q=}")
    print(f"{qdot=}")
    print(f"{spl(times[-1] + control_period, 1)=}")
    print(f"{spl(times[-1] + control_period, 2)=}")

    u = controller.control_step(
        q=q,
        qdot=qdot,
        q_d=q_final,
        qdot_d=spl(control_period, 1),
        qddot_d=spl(control_period, 2)
    ) 

    print(f"{u=}")
    motors.set_pwm(u)

    # Helps while loop run at a fixed frequency
    loop_manager.sleep()
    # time.sleep(control_period)

motors._torque_enable(0)

# Extract results
time_stamps = np.asarray(times)
joint_positions = np.rad2deg(joint_positions).T

print(f"{time_stamps=}")
print(f"{joint_positions=}")

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