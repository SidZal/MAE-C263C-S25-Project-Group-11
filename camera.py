import cv2 as cv
import numpy as np
import imutils
import time

def rolling_add(arr, obj):
    arr = np.delete(arr, 0, axis=0)
    arr = np.append(arr, obj, axis=0)

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

        self.height = int(self.cam.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.width = int(self.cam.get(cv.CAP_PROP_FRAME_WIDTH))

        # Init kinematics
        rolling_range = 5
        self.ball_pos = np.asarray([self.width/2, self.height/2], dtype=int)
        self.ball_vel = np.zeros((rolling_range, 2))
        self.ball_radius_measured = None
        self.ball_predicted_path = None
        self.clear_predicted_path()

        self.ball_endpoint_rolling = np.zeros((rolling_range, 2), dtype=int)
        self.ball_endpoint = None

    def clear_predicted_path(self):
        self.ball_predicted_path = np.zeros((0, 1), dtype = int)

    def find_ball(self):
        """
        Grabs camera frame and does computer vision
            :return: success (T/F) 
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
                    new_vel = np.array([[(int(x) - self.ball_pos[0]) / dt, (int(y) - self.ball_pos[1]) / dt]])
                    
                    rolling_add(self.ball_vel, new_vel)

                    self.last_time = loop_time

                    # update ball pos/rad
                    self.ball_pos = [int(x), int(y)]
                    self.ball_radius_measured = int(radius)
                else:
                    self.ball_pos.clear()
                
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
        [bx, by] = self.ball_pos

        # Take rolling average
        [vx, vy] = np.mean(self.ball_vel, axis=0)

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
            self.clear_predicted_path()

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
                self.ball_predicted_path = np.append(self.ball_predicted_path, [[bx, by, bounds_violation]], axis=0)

            # TODO: Calculate exact cross point
            if np.size(self.ball_predicted_path) == 0:
                self.ball_endpoint = None
            else:
                rolling_add(self.ball_endpoint_rolling, [self.ball_predicted_path[-1]])
                self.ball_endpoint = np.mean(self.ball_predicted_path, axis=0)

            return self.ball_endpoint

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

        # Draw circle about smoothed endpoint
        if self.ball_endpoint is not None:
            cv.circle(self.frame, self.ball_endpoint, self.ball_radius_measured, self.color[1], 2)
            cv.circle(self.frame, self.ball_pos, 5, (0, 0, 255), -1)

        # draw velocity vector
        velScalar = .2
        pointTo = (int(self.ball_pos[0] + velScalar*self.ball_vel[0]), int(self.ball_pos[1] + velScalar*self.ball_vel[1]))
        cv.arrowedLine(self.frame, self.ball_pos, pointTo, (0, 255, 0), 2)

        # Draw prediction
        for prediction_point in self.ball_predicted_path:
            clr = (255, 0, 0)
            if prediction_point[2] != 0:
                clr = (0, 0, 255) # draw bounces red
            
            cv.circle(self.frame, prediction_point[0:2], self.ball_radius, color=clr, thickness=1)
            cv.drawMarker(self.frame, prediction_point[0:2], color=(255, 255, 255), markerType=cv.MARKER_CROSS, thickness=1)

        # Show playback
        cv.imshow("Camera", self.frame)

        # kill key q
        return not cv.waitKey(1) == ord('q')

