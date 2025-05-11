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
        ball_kinematics = []

        bx = ball_pos[0]
        by = ball_pos[1]

        vx = ball_vel[0] # should always be < 0
        vy = ball_vel[1]
        # Radius to watch for cross
        r_prep = espr*self.reach

        # Absolute y max given arena bounds
        y_max = arena_width/2

        # Ensure ball is coming towards robot with significant velocity
        if (vx < 0) and (vx**2 + vy**2 > 0.1**2):
            # Initial angle of velocity
            theta = math.atan2(vy, vx)
            print(theta)
            print()

            # Iterate until ball is past arm or within radius
            while bx**2 + by**2 >  r_prep**2 and bx > 0:
                # Calculate incoming ball position differentials
                dx = vx*dt
                dy = vy*dt
                print(math.atan2(dy, dx))

                bounds_violation = 0

                # Check for upper bound violation
                if by + dy + ball_radius > y_max:
                    bounds_violation = 1

                # Check for lower bound violation
                if by + dy - ball_radius < -y_max:
                    bounds_violation = -1

                if bounds_violation != 0:
                    pre_collision_dy = bounds_violation * (y_max - ball_radius) - by
                    print("dypre", pre_collision_dy)

                    post_collision_dy = dy - pre_collision_dy
                    print("dypost", post_collision_dy)

                    dy = dy - 2*post_collision_dy
                    print("dy", dy)
                    print()

                    vy *= -1
                bx += dx
                by += dy
                ball_kinematics.append([bx, by, bounds_violation, vx, vy])

            # TODO: Calculate exact cross point

        return np.asarray(ball_kinematics) # exact cross point, and maybe array for vis?
