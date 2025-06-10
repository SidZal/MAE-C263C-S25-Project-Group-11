import numpy as np
from numpy.typing import NDArray
from typing import Tuple

# Project modules
from fixedFrequencyLoopManager import fixedFrequencyLoopManager

# Inverse Dynamics (inertia only) controller for planar 2R robot
# Based on C263C hw#4
class stepController:
    def __init__(
        self,
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
        link_length: Tuple[float],
        link_mass: Tuple[float],
        link_inertia: Tuple[float],
        motor_mass: Tuple[float],
        motor_inertia: Tuple[float],
        gear_ratio: Tuple[int],
        controller: str # Controller selection
    ):
        '''
        Controller Options:
        'Simplified Inverse Dynamics'
        'Inertial Inverse Dynamics'
        'Inverse Dynamics'
        '''
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

        # Average Inertia Matrix
        self.B_avg = np.zeros((2, 2))
        self.B_avg[0, 0] = self.I_l1 + self.m_l1*self.l_1**2 + self.k_r1**2*self.I_m1 + self.I_l2 + self.m_l2*(self.a_1**2 + self.l_2**2 + 2*self.a_1*self.l_2)
        self.B_avg[1, 1] = self.I_l2 + self.m_l2*self.l_2**2 + self.k_r2**2*self.I_m2
        
        # Full Inertial Matrix: Initialize with constants
        self.Bq = np.zeros((2, 2))
        self.Bq[0, 0] = self.I_l1 + self.m_l1*self.l_1**2 + self.k_r1**2*self.I_m1 + self.I_l2 + self.I_m2 + self.m_m2*self.a_1**2
        self.Bq[1, 0] = self.Bq[0, 1] = self.I_l2 + self.k_r2*self.I_m2 + self.m_l2*self.l_2**2
        self.Bq[1, 1] = self.I_l2 + self.m_l2*self.l_2**2 + self.k_r2**2*self.I_m2

        if controller == 'Simplified Inverse Dynamics':
            # Simplified Inverse Dynamics: Similar to C263C HW#4
            self.control_step = self.simplified_inverse_dynamics_step

        elif controller == 'Inertial Inverse Dynamics':
            # Inverse Dynamics, neglect centrifugal and coriolis forces
            self.control_step = self.inertial_inverse_dynamics_step

        elif controller == 'Inverse Dynamics':
            # Inverse Dynamics as taught in class
            self.control_step = self.inverse_dynamics_step

        else:
            raise ValueError("Invalid Controller Selected!")

    def _compute_inertial_matrix(self, q2: np.double):
        # Calculate configuration-specific terms
        config_terms_Bq = np.zeros((2, 2))
        config_terms_Bq[0, 0] = self.m_l2*(self.a_1**2 + self.l_2**2 + 2*self.a_1*self.l_2*np.cos(q2))
        config_terms_Bq[1, 0] = config_terms_Bq[0, 1] = self.m_l2*self.a_1*self.l_2*np.cos(q2)

        # Sum constants and configuration-specific terms
        return config_terms_Bq + self.Bq
    
    def _compute_nonlinear_terms(self, q2: np.double, qdot: NDArray[np.double]):
        centrifugal = np.zeros((2,2))
        centrifugal[1, 0] = centrifugal[0, 1] = self.m_l2*self.a_1*self.l_2*np.sin(q2)
        centrifugal[0, 1] *= -1

        coriolis = np.zeros((2))
        coriolis[1] = -2*self.m_l2*self.a_1*self.l_2*np.sin(q2)

        return centrifugal @ (np.asarray(qdot)**2) + coriolis.T * np.prod(qdot)

    def simplified_inverse_dynamics_step(
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

        # can use * instead of @ since it's diagonal anyways
        u = self.B_avg*(qddot_d + self.K_P*q_error + self.K_D*qdot_error)

        return np.diag(u)
        
    def inertial_inverse_dynamics_step(
        self,
        q: NDArray[np.double], # rad
        qdot: NDArray[np.double], # rad/s
        q_d: NDArray[np.double], # rad
        qdot_d: NDArray[np.double], # rad/s
        qddot_d: NDArray[np.double] # rad/s^2 
    ):
        # joint position error
        q_error = q_d - q

        # joint velocity error
        qdot_error = qdot_d - qdot

        # inertia matrix
        Bq = self._compute_inertial_matrix(q[1])

        return Bq @ (qddot_d + self.K_P @ q_error + self.K_D @ qdot_error)
    
    def inverse_dynamics_step(
        self,
        q: NDArray[np.double], # rad
        qdot: NDArray[np.double], # rad/s
        q_d: NDArray[np.double], # rad
        qdot_d: NDArray[np.double], # rad/s
        qddot_d: NDArray[np.double] # rad/s^2 
    ):
        # joint position error
        q_error = q_d - q

        # joint velocity error
        qdot_error = qdot_d - qdot

        y = qddot_d + self.K_P @ q_error + self.K_D @ qdot_error

        # inertia matrix
        Bq = self._compute_inertial_matrix(q[1])

        # non-linear state feedback
        nq = self._compute_nonlinear_terms(q[1], qdot)

        return Bq @ y + nq