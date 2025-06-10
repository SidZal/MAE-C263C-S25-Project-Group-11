import time
import numpy as np
from scipy.interpolate import CubicSpline
from typing import Tuple, List
from numpy.typing import NDArray

from servosChainClass import servos
from stepcontroller import stepController

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
        loop_freq: float, # Frequency at which to loop controller
        arena_constraints: Tuple[tuple], # min and max y constraints due to arena walls
        controller: str,
        homing_offsets: List[int] = [0, 0],
        lost_endpoint_threshold: int = 3,
        traj_padding: float = 1.,
        flick_time: float = 0.15,
        flick_threshold: float = 0.1,
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
        self.controller = stepController(
            K_P=K_P,
            K_D=K_D,
            link_length=link_length,
            link_mass=link_mass,
            link_inertia=link_inertia,
            motor_mass=motor_mass,
            motor_inertia=motor_inertia,
            gear_ratio=gear_ratio,
            controller=controller
        )

        # Class Vars
        # Endpoint Management
        self.lost_endpoint_threshold = lost_endpoint_threshold # after how many loops should bot give up if no endpoints are provided?
        self.goal_line_x = arena_constraints[0][0]
        self.goal_line_y_range = arena_constraints[:][1]
        self._reset_endpoint()

        # Trajectory Management
        self.mode = 0 # Bot's mode management: 0 = Waiting, 1 = Ball Arrival, 2 = Flick Upon Ball Approach
        self.loop_freq = loop_freq
        self.spl = None
        self.traj_padding = traj_padding
        self.flick_time = flick_time # how long to spend flicking
        self.flick_threshold = flick_threshold # how close to start flick? (time)
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
    def update(self, cam_info: tuple):
        # Ignore updates during flick (mode 2)
        if self.mode != 2:
            self.ball_pos = cam_info[0]
            self.ball_endpoint = cam_info[1]
            self.time_to_endpoint = cam_info[2]
            self.ball_vel = cam_info[3]

    # Private endpoint handler for when endpoint is processed
    def _reset_endpoint(self):
        self.time_to_endpoint = None
        self.endpoint_lost_counter = 0

    @property
    def _normal_traj_time(self):
        return max([self.time_to_endpoint, self.traj_padding])

    def _flick_start_condition(self, bot_pos: NDArray[np.double]):
        # meter distance between bot pos and ball pos
        return np.sqrt(np.sum((self.ball_pos - bot_pos)**2)) < self.flick_threshold

    def _get_target_pos(self):
        if self.ball_endpoint[1] > self.goal_line_y_range[1]:
            y = self.goal_line_y_range[1]
        elif self.ball_endpoint[1] < self.goal_line_y_range[0]:
            y = self.goal_line_y_range[0]
        else:
            y = self.ball_endpoint[1]

        return np.asarray((self.goal_line_x, y))

    def step(self):
        q_actual = self.q
        qdot_actual = self.qdot

        fk_pos = self._forward_kinematics(q_actual)
        target_pos = fk_pos

        # Determine course of action
        if self.mode == 0:
            # WAITING Mode: Don't care/know where ball is
            
            # Should we care?
            if self.ball_endpoint is not None:
                # Ball is coming! switch mode and start prep
                self.mode = 1

                print("Generating trajectory")
                target_pos = self._get_target_pos()
                self.spl = self._generate_trajectory(0, self._normal_traj_time, target_pos)

        elif self.mode == 1:
            # PREPARE FOR BALL Mode: Ball is coming to bot
            
            # Check if still coming to bot
            if self.time_to_endpoint:
                # Check if should start flick
                # thresholds: self.time_to_endpoint < self.flick_threshold
                target_pos = self._get_target_pos()

                if self._flick_start_condition(fk_pos):
                    # Start flick!
                    print("Flicking!")
                    self.mode = 2
                    self.dir = np.asarray([.1, 0])

                    self.flick_start = time.time()
                    self.spl = self._generate_trajectory(0, self.flick_time + self.traj_padding, target_pos + self.flick_scale*self.dir)
                    
                else:
                    self.spl = self._generate_trajectory(0, self._normal_traj_time, target_pos)

                self._reset_endpoint()
            else:
                # Lost endpoint: to counter hallucinated endpoints
                self.endpoint_lost_counter += 1
                if self.endpoint_lost_counter > self.lost_endpoint_threshold:
                    self.mode = 0 # back to waiting...
                    self.spl = None

        elif self.mode == 2:
            # FLICK Mode: Hit ball once it is close enough
            print("Flick mode")
            self.dir = np.asarray([.1, 0])

            # Check if flick should end after this
            time_since_flick_start = time.time() - self.flick_start
            flick_end = self.flick_time - time_since_flick_start
            if flick_end <= 0:
                self.mode = 0
            else:
                target_pos = self._get_target_pos()
                self.spl = self._generate_trajectory(0, flick_end + self.traj_padding, target_pos + self.flick_scale*self.dir)
                
        # Control step
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

        return fk_pos, target_pos
    
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
