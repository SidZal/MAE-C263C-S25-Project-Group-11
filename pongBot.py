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
        lost_endpoint_threshold: int = 3,
        traj_padding: float = 0.2,
        flick_time: float = 0.15,
        flick_threshold: float = 0.1,
        flick_scale: float = 2
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

        # Class Vars
        # Endpoint Management
        self.lost_endpoint_threshold = lost_endpoint_threshold # after how many loops should bot give up if no endpoints are provided?
        self._reset_endpoint()

        # Trajectory Management
        self.mode = 0 # Bot's mode management: 0 = Waiting, 1 = Ball Arrival, 2 = Flick Upon Ball Approach
        self.spl = None
        self.traj_padding = traj_padding # time to pad trajectory to endpoint
        self.flick_time = flick_time # how long to spend flicking
        self.flick_threshold = flick_threshold # how close to start flick?
        self.flick_scale = flick_scale # how far to flick to?

    @property
    def q(self):
        return np.asarray(self.motors.read_position(), dtype=np.double)
    
    @property
    def qdot(self):
        return np.asarray(self.motors.read_velocity(), dtype=np.double)

    # External updater for endpoint
    def update_endpoint(self, endpoint: NDArray[np.double], dir: NDArray[np.double], time: float):
        # Ignore updates during flick (mode 2)
        if self.mode is not 2:
            self.endpoint = endpoint
            self.dir = dir
            self.time_to_endpoint = time

    # Private endpoint handler for when endpoint is processed
    def _reset_endpoint(self):
        self.endpoint = self.dir = self.time_to_endpoint = None
        self.endpoint_lost_counter = 0

    def step(self):
        # Determine course of action
        if self.mode is 0:
            # WAITING Mode: Don't care/know where ball is
            
            # Should we care?
            if self.endpoint:
                # Ball is coming! switch mode and start prep
                self.mode = 1
                self.spl = self._generate_trajectory(self.time_to_endpoint + self.traj_padding, self.endpoint)

        elif self.mode is 1:
            # PREPARE FOR BALL Mode: Ball is coming to bot
            
            # Check if still coming to bot
            if self.time_to_endpoint:

                # Check if should start flick
                if self.time_to_endpoint < self.flick_threshold:
                    # Start flick!
                    self.mode = 2

                    self.spl = self._generate_trajectory(self.flick_time, self.endpoint + self.flick_scale*self.dir)
                    self.flick_start = time.time()
                else:
                    self.spl = self._generate_trajectory(self.time_to_endpoint + self.traj_padding, self.endpoint)

                self._reset_endpoint()
            else:
                # Lost endpoint: to counter hallucinated endpoints
                self.endpoint_lost_counter += 1
                if self.endpoint_lost_counter > self.lost_endpoint_threshold:
                    self.mode = 0 # back to waiting...

        elif self.mode is 2:
            # FLICK Mode: Hit ball once it is close enough

            # Check if flick should end after this
            time_since_traj_start = time.time() - self.flick_start
            if time_since_traj_start > self.flick_time:
                self.mode = 0

            self.spl = self._generate_trajectory(self.flick_time - time_since_traj_start, self.endpoint + self.flick_scale*self.dir)
                
        # Control step
        q_actual = self.q
        qdot_actual = self.qdot

        if self.spl:
            time_since_traj_start = time.time() - self.trajectory_start_time
            q_d = self.spl(time_since_traj_start)
            qdot_d = self.spl(time_since_traj_start, 1)
            qddot_d = self.spl(time_since_traj_start, 2)
        else:
            q_d = q_actual
            qdot_d = qdot_actual
            qddot_d = 0
        
        u = self.controller.control_step(
            q=q_actual, 
            qdot=qdot_actual,
            q_d=q_d,
            qdot_d=qdot_d,
            qddot_d=qddot_d
        )

        self.motors.set_pwm(u)

        return self._forward_kinematics(q_actual), self._forward_kinematics(q_d)
    
    def _generate_trajectory(self, end_time: float, end_point: NDArray[np.double]):
        trajectory_end_angles = self._inverse_kinematics(end_point)

        # Prep waypoints for CubicSpline
        time_vias = np.asarray(0, end_time)
        joint_angles_vias = np.asarray(self.q, trajectory_end_angles)
        joint_omegas_vias = ((1, self.qdot), (1, np.zeros(2)))

        self.spl = CubicSpline(x=time_vias, y=joint_angles_vias, axis=1, bc_type=joint_omegas_vias)

        self.trajectory_start_time = time.time()

    def _inverse_kinematics(self, pos: NDArray[np.double], config: int = 1):
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

        # Workspace constraints
        assert(r <= np.sum(self.l) and x > 0)
        
        th1 = np.arccos((self.l[0]**2 + r**2 - self.l[1]**2) / (2*self.l[0]*r)) + config * np.arctan2(y, x)
        th2 = np.pi - np.arccos((self.l[0]**2 + self.l[1]**2 - r**2) / (2*self.l[0]*self.l[1]))

        return np.asarray((th1, th2))

    def _forward_kinematics(self, th: NDArray[np.double]):
        '''
        Planar 2R FK

        :param th: (th1, th2) in rad
        :return pos: [x,y] in m
        '''
        x = self.l[0] * np.cos(th[0]) + self.l[1] * np.cos(th[0] + th[1])
        y = self.l[0] * np.sin(th[0]) + self.l[1] * np.sin(th[0] + th[1])

        return np.asarray((x, y))
