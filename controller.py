import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from dxl import (
    DynamixelMode, 
    DynamixelModel, 
    DynamixelMotorGroup, 
    DynamixelMotorFactory, 
    DynamixelIO
)
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel


class InverseDynamicsController:
    def __init__(
        self,
        motor_group: DynamixelMotorGroup,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        max_duration_s: float = 2.0,
    ):
        # Controller Related Variables
        # ------------------------------------------------------------------------------
        self.q_initial_rad = np.deg2rad(q_initial_deg)
        self.q_desired_rad = np.deg2rad(q_desired_deg)

        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)

        self.control_freq_Hz = 30.0
        self.max_duration_s = float(max_duration_s)
        self.control_period_s = 1 / self.control_freq_Hz
        self.loop_manager = FixedFrequencyLoopManager(self.control_freq_Hz)
        self.should_continue = True

        self.joint_position_history = deque()
        self.time_stamps = deque()
        # ------------------------------------------------------------------------------

        # Manipulator Parameters
        # ------------------------------------------------------------------------------
        self.a_1 = 0.15 # m
        self.a_2 = 0.125 # m
        self.l_1 = self.a_1 / 2 # m
        self.l_2 = self.a_2 / 2 # m
        self.m_l1 = self.m_l2 = 0.035 # kg
        self.I_l1 = self.I_l2 = 0.001 # kg-m^2
        self.m_m1 = self.m_m2 = 0.08 # kg
        self.I_m1 = self.I_m2 = 0.007 # kg-m^2
        self.k_r1 = self.k_r2 = 193
        # ------------------------------------------------------------------------------

        # Inertia Matrix
        # ------------------------------------------------------------------------------
        self.B_avg = np.zeros((2, 2))
        self.B_avg[0, 0] = self.I_l1 + self.m_l1*self.l_1**2 + self.k_r1**2*self.I_m1 + self.I_l2 + self.m_l2*(self.a_1**2 + self.l_2**2 + 2*self.a_1*self.l_2)
        self.B_avg[1, 1] = self.I_l2 + self.m_l2*self.l_2**2 + self.k_r2**2*self.I_m2
        # ------------------------------------------------------------------------------

        # DC Motor Modeling
        # ------------------------------------------------------------------------------
        self.motor_group: DynamixelMotorGroup = motor_group
        self.pwm_limits = []
        for info in self.motor_group.motor_info.values():
            self.pwm_limits.append(info.pwm_limit)
        self.pwm_limits = np.asarray(self.pwm_limits)
        self.motor_model = DCMotorModel(
            self.control_period_s, pwm_limits=self.pwm_limits
        )
        # ------------------------------------------------------------------------------

        # Clean Up / Exit Handler Code
        # ------------------------------------------------------------------------------
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        # ------------------------------------------------------------------------------
        
    def start_control_loop(self):

        start_time = time.time()
        while self.should_continue:

            # Read joint position feedback and convert resulting dict into NumPy array
            q_rad = np.asarray(list(self.motor_group.angle_rad.values()))

            # Read joint velocity feedback and convert resulting dict into NumPy array
            qdot_rad_per_s = (
                np.asarray(list(self.motor_group.velocity_rad_per_s.values()))
            )

            self.joint_position_history.append(q_rad)  # Save for plotting
            self.time_stamps.append(time.time() - start_time)  # Save for plotting

            # Check termination criterion
            if self.time_stamps[-1] - self.time_stamps[0] > self.max_duration_s:
                self.stop()
                return

            # Compute joint position error
            q_error = self.q_desired_rad - q_rad

            # Compute desired cubic spline trajectory
            dt = self.control_period_s
            times = np.arange(0, self.max_duration_s + dt, dt)
            waypoint_times = np.asarray([0, self.max_duration_s / 2, self.max_duration_s])
            waypoints = np.stack([self.q_initial_rad, (self.q_initial_rad + self.q_desired_rad)/2, self.q_desired_rad], axis=1)

            q_d_traj, qdot_d_traj, qddot_d_traj = self.eval_cubic_spline_traj(
            times=times, waypoint_times=waypoint_times, waypoints=waypoints
            )

            # Compute joint velocity error
            qdot_error = qdot_d_traj - qdot_rad_per_s

            # Calculate control action
            u = self.B_avg*qddot_d_traj + self.K_P*q_error + self.K_D*qdot_error

            # Convert torque control action into a PWM command using model of Dynamixel motors
            pwm_command = self.motor_model.calc_pwm_command(u)

            # Sending joint PWM commands 
            self.motor_group.pwm = {
                dxl_id: pwm_value
                for dxl_id, pwm_value in zip(
                    self.motor_group.dynamixel_ids, pwm_command, strict=True
                )
            }

            # Helps while loop run at a fixed frequency
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motor_group.disable_torque()

    def signal_handler(self, *_):
        self.stop()

    def eval_cubic_spline_traj(
        self,
        times: NDArray[np.double],
        waypoint_times: NDArray[np.double],
        waypoints: NDArray[np.double],
    ) -> tuple[NDArray[np.double], NDArray[np.double], NDArray[np.double]]:
        times = np.asarray(times, dtype=np.double)
        waypoint_times = np.asarray(waypoint_times, dtype=np.double)
        waypoints = np.asarray(waypoints, dtype=np.double)

        spl = CubicSpline(x=waypoint_times, y=waypoints, axis=1, bc_type="clamped")

        return spl(times), spl(times, 1), spl(times, 2)

    def go_to_home_configuration(self):
        self.should_continue = True
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.Position)
        self.motor_group.enable_torque()

        # Move to home position (currently self.q_initial)
        home_positions_rad = {
            dynamixel_id: self.q_initial_rad[i]
            for i, dynamixel_id in enumerate(self.motor_group.dynamixel_ids)
        }
        
        self.motor_group.angle_rad = home_positions_rad
        time.sleep(0.5)
        abs_tol = math.radians(1.0)
        
        should_continue_loop = True
        while should_continue_loop:
            should_continue_loop = False
            q_rad = self.motor_group.angle_rad
            for dxl_id in home_positions_rad:
                if abs(home_positions_rad[dxl_id] - q_rad[dxl_id]) > abs_tol:
                    should_continue_loop = True
                    break
            
        # Set PWM Mode (i.e. voltage control)
        self.motor_group.disable_torque()
        self.motor_group.set_mode(DynamixelMode.PWM)
        self.motor_group.enable_torque()


if __name__ == "__main__":

    # Motor Positions
    # ------------------------------------------------------------------------------
    # Leftmost Configuration = [335, 240]
    # Center Configuration = [300, 220]
    # Rightmost Configuration = [255, 245]
    # ------------------------------------------------------------------------------

    # Gain Matrices
    K_P = np.diag([ , ])
    K_D = np.diag([ , ])
    
    # Create `DynamixelIO` object to store the serial connection to U2D2
    dxl_io = DynamixelIO(
        device_name="COM3",
        baud_rate=57_600,
    )

    # Create `DynamixelMotorFactory` object to create dynamixel motor object
    motor_factory = DynamixelMotorFactory(
        dxl_io=dxl_io,
        dynamixel_model=DynamixelModel.MX28
    )

    # Create Dynamixel IDs
    dynamixel_ids = [1, 2]
    motor_group = motor_factory.create(*dynamixel_ids)

    while # Condition TBD

        # Get joint angles
        q_initial = np.asarray(list(motor_group.angle_rad.values()))
        q_desired = # Get from ball position
        
        # Make controller
        controller = InverseDynamicsController(
            motor_group=motor_group,
            K_P=K_P,
            K_D=K_D,
            q_initial_deg=q_initial,
            q_desired_deg=q_desired
        )

        # Run controller
        controller.start_control_loop()

        # Insert code to hit the ball


"""
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
    fig.savefig(fig_file_name)
    # ----------------------------------------------------------------------------------
    plt.show()
"""
