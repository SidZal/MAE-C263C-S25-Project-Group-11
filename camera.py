import cv2 as cv
import numpy as np
import imutils
import time

def rolling_add(arr, obj):
    arr = np.delete(arr, 0, axis=0)
    arr = np.append(arr, [obj], axis=0)
    return arr

class cameraModule:
    """
    Class for handling camera and cv calls
    """
    def __init__(self, capture, color_bounds, arena_height, ball_radius, position_rolling_mean_range = 3, velocity_rolling_mean_range = 5, endpoint_rolling_mean_range = 5):
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
        self.ball_pos_rolling = np.zeros((position_rolling_mean_range, 2), dtype=int)
        self.ball_pos = None

        self.ball_radius_measured = None

        self.ball_vel_rolling = np.zeros((velocity_rolling_mean_range, 2))
        self.ball_vel = None
        
        self.ball_predicted_path = None
        self.clear_predicted_path()

        self.ball_endpoint_rolling = np.zeros((endpoint_rolling_mean_range, 2), dtype=int)
        self.ball_endpoint = None

    def clear_predicted_path(self):
        self.ball_predicted_path = np.zeros((0, 3), dtype = int)

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
                    # Moments - unnecessary?
                    # M = cv.moments(c)
                    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Old pos for velocity calc
                    last_pos = self.ball_pos

                    # Update ball position
                    self.ball_pos_rolling = rolling_add(self.ball_pos_rolling, np.array((int(x), int(y)), dtype=int))
                    self.ball_pos = np.mean(self.ball_pos_rolling, axis=0, dtype=int)

                    loop_time = time.perf_counter_ns()

                    # Calculate instantaneous ball velocity
                    if last_pos is not None:
                        dt = (loop_time - self.last_time) / (1e9) # convert to s
                        new_vel = np.array((self.ball_pos - last_pos) / dt)

                        self.ball_vel_rolling = rolling_add(self.ball_vel_rolling, new_vel)
                        self.ball_vel = np.mean(self.ball_vel_rolling, axis=0)
                    
                    self.last_time = loop_time        

                    self.ball_radius_measured = int(radius)
                else:
                    self.ball_pos = None
                
        return ret

    def predict_path(self, espr = 0.8, dt = 0.5):
        """
        Casts ray from ball position w/ velocity vector
        to determine where ball will cross into some radius around
        the robot's base frame
        
        Assumes frictionless rolling and momentum preserved on collision

        :param epsr: epsilon r, % of reach to watch for ball cross
        :param dt: delta time, time resolution
        """
        if self.ball_pos is not None and self.ball_vel is not None:
            [bx, by] = self.ball_pos

            # Take rolling average
            [vx, vy] = self.ball_vel

            # Radius to watch for cross
            r_prep = espr*1

            # Absolute y max given arena bounds
            y_max = self.arena_height/2
            y_mid = self.height/2

            # Reset ball path
            self.clear_predicted_path()

            # Ensure ball is coming towards robot with significant velocity
            if (vx < 0) and (vx**2 + vy**2 > 50**2):
                # TODO improve this loop termination condition: where to predict until?
                while bx > 50:
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
                        # Collision change calculation
                        pre_collision_dy = y_mid + bounds_violation * (y_max - self.ball_radius) - by
                        post_collision_dy = dy - pre_collision_dy
                        dy = dy - 2*post_collision_dy

                        # Flip velocity after collision
                        vy *= -1
                    
                    # New pos
                    bx = int(bx + dx)
                    by = int(by + dy)

                    # Store prediction step
                    self.ball_predicted_path = np.append(self.ball_predicted_path, np.array([[bx, by, bounds_violation]]), axis=0)

                # TODO: Improve determination of desired position rather than just the end of the predicted path
                if np.size(self.ball_predicted_path) == 0:
                    self.ball_endpoint = None
                else:
                    self.ball_endpoint_rolling = rolling_add(self.ball_endpoint_rolling, self.ball_predicted_path[-1][0:2])
                    self.ball_endpoint = np.mean(self.ball_endpoint_rolling, axis=0, dtype=int)

        return self.ball_endpoint

    def playback(self):
        '''
        Optional playback to show ball tracking and path prediction
        '''
        # Draw arena
        cv.line(self.frame, (0, int(self.height/2 + self.arena_height/2)), (self.width, int(self.height/2 + self.arena_height/2)), 0, 4)
        cv.line(self.frame, (0, int(self.height/2 - self.arena_height/2)), (self.width, int(self.height/2 - self.arena_height/2)), 0, 4)

        if self.ball_pos is not None:
            # Draw circle about ball 
            cv.circle(self.frame, self.ball_pos, self.ball_radius_measured, self.color[1], 2)
            cv.circle(self.frame, self.ball_pos, 5, (0, 0, 255), -1)

            # Draw circle about smoothed endpoint
            if self.ball_endpoint is not None:
                cv.circle(self.frame, self.ball_endpoint, self.ball_radius_measured, self.color[1], 2)
                cv.circle(self.frame, self.ball_endpoint, 5, (0, 0, 255), -1)

            # draw velocity vector
            if self.ball_vel is not None:
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

