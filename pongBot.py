import time
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple
from numpy.typing import NDArray

from servosChainClass import servos
from stepcontroller import inverseDynamicsControl

class pongBot:
    def __init__(
        self,
        motors_port: str,
        link_length: Tuple[float],
        link_mass: Tuple[float],
        link_inertia: Tuple[float],
        motor_mass: Tuple[float],
        motor_inertia: Tuple[float],
        gear_ratio: Tuple[int],
        K_P: NDArray[np.double],
        K_D: NDArray[np.double],
    ):
        # Manipulator Parameters
        self.l = link_length # m
        self.ml = link_mass # kg
        self.Il = link_inertia # kg*m^2
        self.mm = motor_mass # kg
        self.Im = motor_inertia # kg*m^2
        self.kr = gear_ratio

        # Project Modules
        self.motors = servos(port=motors_port, num_motors=2)
        self.controller = inverseDynamicsControl(
            K_P=K_P,
            K_D=K_D,
            link_length=link_length,
            link_mass=link_mass,
            link_inertia=link_inertia,
            motor_mass=motor_mass,
            motor_inertia=motor_inertia,
            gear_ratio=gear_ratio
        )

        # Class Trakcers
        self.endpoint = None
        self.on_traj = False 
        self.spl = None
        self.time_along_current_traj = 0
        self.traj_

    @property
    def q(self):
        return np.asarray(self.motors.read_position(), dtype=np.double)
    
    @property
    def qdot(self):
        return np.asarray(self.motors.read_velocity(), dtype=np.double)

    # External updater for endpoint
    def update_endpoint(self, endpoint: Tuple[float]):
        self.endpoint = endpoint

    # TODO: finish step function
    def step(self):
        # Determine course of action
        if self.on_traj:
            # If ball is past some threshold, flick! (new trajectory)
            if 0:
                ...
            else:
                # Regenerate traj
                ...
        else:
            # Ball not found/approaching -> do nothing
            q_d = self.q
            qdot_d = qdot_d
            qdot_d = np.zeros(2)

            if self.endpoint:
                self.trajectory_end_angles = self._inverse_kinematics(self.endpoint)
                self._generate_trajectory()
                
        # control step
        self.time_since_traj_start = time.time() - self.current_traj_start
        self.controller.control_step(
            q=self.q, 
            qdot=self.qdot,
            q_d=q_d,
            qdot_d=qdot_d,
            qddot_d=qdot_d
        )

    def _generate_trajectory(self):
        time_vias = np.asarray(..., ...)
        joint_angles_vias = np.asarray(self.q, self.trajectory_end_angles)
        joint_omegas_vias = ((1, self.qdot), (1, np.zeros(2)))
        self.spl = CubicSpline(x=time_vias, y=joint_angles_vias, axis=1, bc_type=joint_omegas_vias)

    def _inverse_kinematics(self, pos: Tuple[float], config: int = 1):
        '''
        Planar 2R IK

        :param pos: (x,y) in m
        :param elbow: 1 -> up (default), -1 -> down
        :return th: [th1, th2] in rad
        '''
        assert(config in [-1, 1])

        x = pos[0]
        y = pos[1]

        r = np.sqrt(x**2 + y**2)
        
        th1 = np.arccos((self.l[0]**2 + r**2 - self.l[1]**2) / (2*self.l[0]*r)) + config * np.arctan2(y, x)
        th2 = np.pi - np.arccos((self.l[0]**2 + self.l[1]**2 - r**2) / (2*self.l[0]*self.l[1]))

        return np.asarray((th1, th2))


    def _forward_kinematics(self, th: Tuple[float]):
        '''
        Planar 2R FK

        :param th: (th1, th2) in rad
        :return pos: (x,y) in m
        '''
        x = self.l[0] * np.cos(th[0]) + self.l[1] * np.cos(th[0] + th[1])
        y = self.l[0] * np.sin(th[0]) + self.l[1] * np.sin(th[0] + th[1])

        return (x, y)
