import math
import signal
import time
from collections import deque
from collections.abc import Sequence
from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from scipy.interpolate import CubicSpline

from mechae263C_helpers.minilabs import FixedFrequencyLoopManager, DCMotorModel
from servosChainClass import servos


class InverseDynamicsController:
    def __init__(
        self,
        motors: servos,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        q_initial_deg: Sequence[float],
        q_desired_deg: Sequence[float],
        max_duration_s: float = 2.0,
    ):
        # Controller Related Variables
        # ------------------------------------------------------------------------------
        self.q_initial_rad = motors.read_position()
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

        self.motors = motors

        # Compute desired cubic spline trajectory
        dt = self.control_period_s
        times = np.arange(0, self.max_duration_s + dt, dt)
        waypoint_times = np.asarray([0, self.max_duration_s / 2, self.max_duration_s])
        waypoints = np.stack([self.q_initial_rad, (self.q_initial_rad + self.q_desired_rad)/2, self.q_desired_rad], axis=1)

        self.q_d_traj, self.qdot_d_traj, self.qddot_d_traj = self.eval_cubic_spline_traj(
        times=times, waypoint_times=waypoint_times, waypoints=waypoints
        )

        # Clean Up / Exit Handler Code
        # ------------------------------------------------------------------------------
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        # ------------------------------------------------------------------------------
        
    def start_control_loop(self):

        start_time = time.time()
        while self.should_continue:

            # Read joint position feedback and convert resulting dict into NumPy array
            q_rad = np.asarray(self.motors.read_position())

            # Read joint velocity feedback and convert resulting dict into NumPy array
            qdot_rad_per_s = (
                np.asarray(self.motors.read_velocity())
            )

            self.joint_position_history.append(q_rad)  # Save for plotting
            self.time_stamps.append(time.time() - start_time)  # Save for plotting

            # Check termination criterion
            elapsed = self.time_stamps[-1] - self.time_stamps[0]
            print(f"{elapsed=}")
            if elapsed > self.max_duration_s - 1:
                self.stop()
                return

            # Compute joint position error
            q_error = self.q_desired_rad - q_rad
            print(f"{q_error=}")

            # Compute desired cubic spline trajectory
            dt = self.control_period_s
            times = np.arange(elapsed, self.max_duration_s + dt, dt)
            waypoint_times = np.asarray([elapsed, (self.max_duration_s+elapsed) / 2, self.max_duration_s])
            waypoints = np.stack([q_rad, (q_rad + self.q_desired_rad)/2, self.q_desired_rad], axis=1)
            print(f"{waypoint_times=}")

            self.q_d_traj, self.qdot_d_traj, self.qddot_d_traj = self.eval_cubic_spline_traj(
            times=times, waypoint_times=waypoint_times, waypoints=waypoints
            )

            # index_traj = int(elapsed/self.max_duration_s * np.shape(self.q_d_traj)[1])
            index_traj = 1
            print(f"{index_traj=}")
            # print(f"{q_d_traj=}")
            # print(f"{qdot_d_traj=}")
            # print(f"{qddot_d_traj=}")

            # Compute joint velocity error
            qdot_error = self.qdot_d_traj[:, index_traj] - qdot_rad_per_s
            print(f"{self.qdot_d_traj[:, index_traj]=}")
            print(f"{self.qddot_d_traj[:, index_traj]=}")


            # Calculate control action
            print(f"{self.B_avg=}")
            u = self.B_avg*(self.qddot_d_traj[:, index_traj] + self.K_P*q_error + self.K_D*qdot_error)
            # u = [[0, 0], [0, 0]]

            print(f"{u=}")
            self.motors.set_pwm(np.diag(u))

            # Helps while loop run at a fixed frequency
            self.loop_manager.sleep()

        self.stop()

    def stop(self):
        self.should_continue = False
        time.sleep(2 * self.control_period_s)
        self.motors._torque_enable(0)

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

        # fig = plt.figure(figsize=(15, 5))
        # ax1 = fig.add_subplot(131)
        # ax2 = fig.add_subplot(132)
        # ax3 = fig.add_subplot(133)
        # ax1.plot(times, spl(times)[0])
        # ax1.plot(times, spl(times)[1])
        # ax2.plot(times, spl(times,1)[0])
        # ax2.plot(times, spl(times,1)[1])
        # ax3.plot(times, spl(times,2)[0])
        # ax3.plot(times, spl(times,2)[1])

        # fig.show()

        return spl(times), spl(times, 1), spl(times, 2)