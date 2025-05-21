import cv2 as cv
import numpy as np
import imutils
import time

class cameraModule:
    """
    Class for handling camera and cv calls
    """
    def __init__(self, capture, color_bounds, arena_height, ball_radius):
        """
        :param capture: opencv capture port, usually 0
        :param color_bounds: 2D tuple of 2 HSV values
        :param arena_height: height of arena
        :param ball_radius: radius of ball
        """
        try:
            self.cam = cv.VideoCapture(capture)
        except:
            print("No camera found.")
        self.color = color_bounds
        
        self.arena_height = arena_height
        self.ball_radius = ball_radius

        self.frame = None
        self.last_time = time.perf_counter_ns()

        # Init kinematics
        self.ball_pos = np.array([300, 300])
        self.ball_vel = np.array([0, 0])
        self.ball_radius_measured = None
        self.ball_path = []

        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))

    def find_ball(self):
        """
        Grabs camera frame and does computer vision
            :return: success (T/F) 
            :return coords: either None or 2x1 x,y in pixels (?)
        """
        ret, frame = self.cam.read()

        if ret:
            # Perform CV
            # resize, blur, convert to HSV
            self.frame = imutils.resize(frame, width=600)
            blurred = cv.GaussianBlur(self.frame, (11, 11), 0)
            hsv = cv.cvtColor(blurred, cv.COLOR_BGR2HSV)

            # Mask: convert to black and white image
            mask = cv.inRange(hsv, self.color[0], self.color[1])
            mask = cv.erode(mask, None, iterations=2)
            mask = cv.dilate(mask, None, iterations=2)

            # Find largest contour: largest swath of white mask
            cntrs = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cntrs = imutils.grab_contours(cntrs)
            center = None

            if len(cntrs) > 0:
                c = max(cntrs, key = cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(c)

                if radius > 10:
                    # might not be neccessary?
                    M = cv.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                    # update ball velocity TODO: a filter
                    loop_time = time.perf_counter_ns()
                    dt = (loop_time - self.last_time) / (1e9) # convert to s
                    self.ball_vel = ( (int(x) - self.ball_pos[0]) / dt, (int(y) - self.ball_pos[1]) / dt)
                    self.last_time = loop_time

                    # update ball pos/rad
                    self.ball_pos = np.array([int(x), int(y)])
                    self.ball_radius_measured = int(radius)
                
            return True
        return False

    def predict_path(self, espr = 0.8, dt = 0.5):
        """
        Casts ray from ball position w/ velocity vector
        to determine where ball will cross into some radius around
        the robot's base frame
        
        Assumes frictionless rolling and momentum preserved on collision

        :param epsr: epsilon r, % of reach to watch for ball cross
        :param dt: delta time, time resolution
        """
        bx = self.ball_pos[0]
        by = self.ball_pos[1]

        vx = self.ball_vel[0] # should always be < 0
        vy = self.ball_vel[1]

        # Radius to watch for cross
        r_prep = espr*1

        # Absolute y max given arena bounds
        y_max = self.arena_height/2
        y_mid = self.height/2

        # Ensure ball is coming towards robot with significant velocity
        if (vx < 0) and (vx**2 + vy**2 > 0.1**2):
            # Initial angle of velocity
            theta = np.atan2(vy, vx)
            # print(theta)

            # reset ball path
            self.ball_path.clear()

            # TODO define where to predict until
            while bx > 0:
                # Calculate incoming ball position differentials
                dx = vx*dt
                dy = vy*dt
                # print(np.atan2(dy, dx))

                bounds_violation = 0

                # Check for upper bound violation
                if by + dy + self.ball_radius > y_mid + y_max:
                    bounds_violation = 1

                # Check for lower bound violation
                if by + dy - self.ball_radius < y_mid - y_max:
                    bounds_violation = -1

                # Redirect in case of bounds violation
                if bounds_violation != 0:
                    pre_collision_dy = y_mid + bounds_violation * (y_max - self.ball_radius) - by
                    # print("dypre", pre_collision_dy)

                    post_collision_dy = dy - pre_collision_dy
                    # print("dypost", post_collision_dy)

                    dy = dy - 2*post_collision_dy
                    # print("dy", dy)

                    vy *= -1
                
                # New pos
                bx = int(bx + dx)
                by = int(by + dy)

                # Store ball this iter
                self.ball_path.append(np.array([bx, by, bounds_violation]))

            # TODO: Calculate exact cross point

    def playback(self):
        '''
        Optional playback to show ball tracking and path prediction
        '''
        # Draw arena
        cv.line(self.frame, (0, int(self.height/2 + self.arena_height/2)), (self.width, int(self.height/2 + self.arena_height/2)), 0, 4)
        cv.line(self.frame, (0, int(self.height/2 - self.arena_height/2)), (self.width, int(self.height/2 - self.arena_height/2)), 0, 4)

        # Draw circle about ball 
        cv.circle(self.frame, self.ball_pos, self.ball_radius_measured, self.color[1], 2)
        cv.circle(self.frame, self.ball_pos, 5, (0, 0, 255), -1)

        # draw velocity vector
        velScalar = .2
        pointTo = (int(self.ball_pos[0] + velScalar*self.ball_vel[0]), int(self.ball_pos[1] + velScalar*self.ball_vel[1]))
        cv.arrowedLine(self.frame, self.ball_pos, pointTo, (0, 255, 0), 2)

        # Draw prediction
        for prediction_point in self.ball_path:
            clr = (255, 0, 0)
            if prediction_point[2] != 0:
                clr = (0, 0, 255) # draw bounces red
            
            cv.circle(self.frame, prediction_point[0:2], self.ball_radius, color=clr, thickness=1)
            cv.drawMarker(self.frame, prediction_point[0:2], color=(255, 255, 255), markerType=cv.MARKER_CROSS, thickness=1)

        # Show playback
        cv.imshow("Camera", self.frame)

        # kill key q
        return not cv.waitKey(1) == ord('q')

