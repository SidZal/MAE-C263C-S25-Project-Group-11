import math
import numpy as np

class pongBot:
    def __init__(self, link_lengths):
        """
        :param link_lengths: 3x1 array of link lengths, including motor 3 to paddle
        """
        self.L = link_lengths
        self.reach = sum(self.L)

    def inverse_kinematics(self):
        ...

    def forward_kinematics(self):
        ...

    def ball_raycast(self, arena_width, ball_radius, ball_pos, ball_vel, espr = 0.8, dt = 0.5):
        """
        Casts ray from ball position w/ velocity vector
        to determine where ball will cross into some radius around
        the robot's base frame
        
        Assumes frictionless rolling and momentum preserved on collision

        :param arena_width: distance from wall to wall
        :param ball_radius: Radius of ball
        :param pos_ball: 2x1, ball x,y in m
        :param pos_ball: 2x1, ball vx,vy in m/s
        :param epsr: epsilon r, % of reach to watch for ball cross
        :param dt: delta time, time resolution
        """
        # TODO: check to ensure some minimum velocity AND ball is coming towards THIS arm
        ...

        bx = ball_pos[0]
        by = ball_pos[1]

        vx = ball_vel[0] # should always be < 0
        vy = ball_vel[1]

        # Radius to watch for cross
        r_prep = espr*self.reach

        # Absolute y max given arena bounds
        y_max = arena_width/2

        # Initial angle of velocity
        theta = math.atan2(vy, vx)

        # Iterate until ball is past arm or within radius
        while bx^2 + by^2 >  (r_prep)^2:
            # Calculate incoming ball position differentials
            dx = vx*dt
            dy = vy*dt

            bounds_violation = 0

            # Check for upper bound violation
            if by + dy + ball_radius > y_max:
                bounds_violation = 1

            # Check for lower bound violation
            if by + dy - ball_radius < -y_max:
                bounds_violation = -1

            if bounds_violation is not 0:
                pre_collision_dy = bounds_violation * (y_max - ball_radius) - by
                pre_collision_dx = pre_collision_dy * math.tan(theta)  

                theta = 2*np.pi - theta
                post_collision_dx = dx - pre_collision_dx

                post_collision_dy = post_collision_dx * math.tan(theta)

                dy = pre_collision_dy - bounds_violation * post_collision_dy

                vy = -vy

            bx += dx
            bx += dy


        # TODO: Calculate exact cross point

        return ... # exact cross point, and maybe array for vis?
