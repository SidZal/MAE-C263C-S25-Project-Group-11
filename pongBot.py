import time
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, List
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
        loop_freq: float,
        homing_offsets: List[int] = [0, 0],
        lost_endpoint_threshold: int = 3,
        traj_padding: float = 1.,
        flick_time: float = 0.15,
        flick_threshold: float = 0.5,
        flick_px_threshold: int = 120,
        flick_scale: float = .15
    ):
        # Manipulator Parameters
        self.l = link_length # m
        self.ml = link_mass # kg
        self.Il = link_inertia # kg*m^2
        self.mm = motor_mass # kg
        self.Im = motor_inertia # kg*m^2
        self.kr = gear_ratio

        # Project Modules
        self.motors = servos(port=motors_port, num_motors=2, homing_offsets=homing_offsets)
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
        self.loop_freq = loop_freq
        self.spl = None
        self.traj_padding = traj_padding
        self.flick_time = flick_time # how long to spend flicking
        self.flick_threshold = flick_threshold # how close to start flick? (time)
        self.flick_px_threshold = flick_px_threshold
        self.flick_scale = flick_scale # how far to flick to?

    @property
    def q(self):
        return np.asarray(self.motors.read_position(), dtype=np.double)
    
    @property
    def qdot(self):
        return np.asarray(self.motors.read_velocity(), dtype=np.double)
    
    def disable_torque(self):
        self.motors._torque_enable(0)

    # External updater for endpoint
    def update_endpoint(
        self, 
        endpoint: NDArray[np.double], 
        dir: NDArray[np.double], 
        time: float, 
        ball_px_x: int,
        ball_pos: NDArray[np.double]
    ):
        # Ignore updates during flick (mode 2)
        if self.mode != 2:
            self.endpoint = endpoint
            self.dir = dir
            self.time_to_endpoint = time
            self.ball_px_x = ball_px_x
            self.ball_pos = ball_pos

    # Private endpoint handler for when endpoint is processed
    def _reset_endpoint(self):
        self.time_to_endpoint = None
        self.endpoint_lost_counter = 0

    @property
    def _normal_traj_time(self):
        return max([self.time_to_endpoint, self.traj_padding])

    def step(self):
        q_actual = self.q
        qdot_actual = self.qdot

        fk_pos = self._forward_kinematics(q_actual)

        # Determine course of action
        if self.mode == 0:
            # WAITING Mode: Don't care/know where ball is
            
            # Should we care?
            if self.endpoint is not None:
                # Ball is coming! switch mode and start prep
                self.mode = 1

                print("Generating trajectory")
                print(f"{self.endpoint=}")
                self.spl = self._generate_trajectory(0, self._normal_traj_time, self.endpoint)

        elif self.mode == 1:
            # PREPARE FOR BALL Mode: Ball is coming to bot
            
            # Check if still coming to bot
            if self.time_to_endpoint:
                # print(f"{self.time_to_endpoint=}")
                print(f"{self.ball_px_x=}")
                # Check if should start flick
                # thresholds: self.time_to_endpoint < self.flick_threshold
                if 20 < self.ball_px_x < self.flick_px_threshold:
                    # Start flick!
                    print("Flicking!")
                    self.mode = 2

                    # self.dir = self.ball_pos[0] - fk_pos
                    # print(f"{self.ball_pos[0]=}")
                    # print(f"{fk_pos=}")
                    # print(f"{self.endpoint=}")
                    # print(f"{self.flick_scale=}")
                    # print(f"{self.dir=}")
                    # print(f"{self.flick_scale*self.dir=}")
                    # print(f"{self.endpoint + self.flick_scale*self.dir=}")
                    self.dir = np.asarray([.1, 0])

                    self.flick_start = time.time()
                    self.spl = self._generate_trajectory(0, self.flick_time + self.traj_padding, self.endpoint + self.flick_scale*self.dir)
                    
                else:
                    self.spl = self._generate_trajectory(0, self._normal_traj_time, self.endpoint)

                self._reset_endpoint()
            else:
                # Lost endpoint: to counter hallucinated endpoints
                self.endpoint_lost_counter += 1
                if self.endpoint_lost_counter > self.lost_endpoint_threshold:
                    self.mode = 0 # back to waiting...

        elif self.mode == 2:
            # FLICK Mode: Hit ball once it is close enough
            print("Flick mode")
            # self.dir = self.ball_pos[0] - fk_pos
            # print(f"{self.ball_pos[0]=}")
            # print(f"{fk_pos=}")
            # print(f"{self.endpoint=}")
            # print(f"{self.dir=}")
            # print(f"{self.flick_scale*self.dir=}")
            # print(f"{self.endpoint + self.flick_scale*self.dir=}")
            self.dir = np.asarray([.1, 0])
            # Check if flick should end after this
            time_since_flick_start = time.time() - self.flick_start
            flick_end = self.flick_time - time_since_flick_start
            if flick_end <= 0:
                self.mode = 0
            else:
                self.spl = self._generate_trajectory(0, flick_end + self.traj_padding, self.endpoint + self.flick_scale*self.dir)
                
        # Control step

        # print(f"{self.spl=}")
        if self.spl:
            q_d = self.q_final
            qdot_d = self.spl(self.loop_freq, 1)
            qddot_d = self.spl(self.loop_freq, 2)
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
        # print(f"{u=}")
        self.motors.set_pwm(u)

        return self._forward_kinematics(q_actual), self.endpoint
    
    def _generate_trajectory(
        self, 
        start_time: float, 
        end_time: float, 
        end_point: NDArray[np.double]
    ):
        
        self.q_final = self._inverse_kinematics(end_point)
        qrad = self.q

        return CubicSpline(
            x=[start_time, (start_time + end_time)/2, end_time],
            y=np.asarray([qrad, (qrad+self.q_final) / 2 ,self.q_final]).T,
            axis=1, 
            bc_type="clamped"
        )

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
        assert(x > 0)

        if r >= np.sum(self.l):
            r = np.sum(self.l)*0.99
        
        th1 = np.arccos((self.l[0]**2 + r**2 - self.l[1]**2) / (2*self.l[0]*r)) + config * np.arctan2(y, x)
        th2 = -config*(np.pi - np.arccos((self.l[0]**2 + self.l[1]**2 - r**2) / (2*self.l[0]*self.l[1])))

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
