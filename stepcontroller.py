import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# Project modules
from fixedFrequencyLoopManager import fixedFrequencyLoopManager

# Inverse Dynamics (inertia only) controller for planar 2R robot
# Based on C263C hw#4
class inverseDynamicsControl:
    def __init__(
        self,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        link_length: Tuple[float],
        link_mass: Tuple[float],
        link_inertia: Tuple[float],
        motor_mass: Tuple[float],
        motor_inertia: Tuple[float],
        gear_ratio: Tuple[int]
    ):
        # Controller Related Variables
        self.K_P = np.asarray(K_P, dtype=np.double)
        self.K_D = np.asarray(K_D, dtype=np.double)

        # Manipulator Parameters
        self.a_1 = link_length[0]
        self.a_2 = link_length[1]
        self.l_1 = self.a_1 / 2
        self.l_2 = self.a_2 / 2
        self.m_l1 = link_mass[0]
        self.m_l2 = link_mass[1]
        self.I_l1 = link_inertia[0]
        self.I_l2 = link_inertia[1]
        self.m_m1 = motor_mass[0]
        self.m_m2 = motor_mass[1]
        self.I_m1 = motor_inertia[0]
        self.I_m2 = motor_inertia[1]
        self.k_r1 = gear_ratio[0]
        self.k_r2 = gear_ratio[1]
        
        # Inertia Matrix
        self.B_avg = np.zeros((2, 2))
        self.B_avg[0, 0] = self.I_l1 + self.m_l1*self.l_1**2 + self.k_r1**2*self.I_m1 + self.I_l2 + self.m_l2*(self.a_1**2 + self.l_2**2 + 2*self.a_1*self.l_2)
        self.B_avg[1, 1] = self.I_l2 + self.m_l2*self.l_2**2 + self.k_r2**2*self.I_m2

    def control_step(
        self, 
        q: NDArray[np.double], # rad
        qdot: NDArray[np.double], # rad/s
        q_d: NDArray[np.double], # rad
        qdot_d: NDArray[np.double], # rad/s
        qddot_d: NDArray[np.double] # rad/s^2
    ):
        # Compute joint position error
        q_error = q_d - q

        # Compute joint velocity error
        qdot_error = qdot_d - qdot

        u = self.B_avg*(qddot_d + self.K_P*q_error + self.K_D*qdot_error)
        return np.diag(u)
        