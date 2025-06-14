import cv2 as cv
import numpy as np
import imutils
import time
from typing import Tuple
from numpy.typing import NDArray

def rolling_add(arr, obj):
    arr = np.delete(arr, 0, axis=0)
    arr = np.append(arr, [obj], axis=0)
    return arr

def rolling_average(arr):
    sum = np.zeros(len(arr[0]))
    for val in arr:
        if (val != np.zeros(len(val), dtype=type(val))).any():
            sum += np.asarray(val)

    return sum / len(arr)

class cameraModule:
    """
    Class for handling camera and cv calls
    """
    def __init__(
        self, 
        capture: int, 
        color_bounds: list[tuple], # list of 2 HSV (tuple) values
        arena_height: int, # px
        ball_radius: int, # px
        endpoint_threshold: int = 50, # px 
        px_per_meter_x: int = 1450, # px/m
        px_per_meter_y: int = 1450, # px/m
        bot_offset: float = 0.08, # m
        height_scale: float = 0.5,
        bot_radius: int = 50,
        position_rolling_mean_range = 3, 
        velocity_rolling_mean_range = 5, 
        endpoint_rolling_mean_range = 5
    ):
        try:
            self.cam = cv.VideoCapture(capture)
        except:
            print("No camera found.")
            time.sleep(5)

        self.color = color_bounds
        
        self.arena_height = arena_height
        self.ball_radius = ball_radius
        self.endpoint_threshold = endpoint_threshold

        self.px_per_meter_x = px_per_meter_x
        self.px_per_meter_y = px_per_meter_y
        self.bot_offset = bot_offset
        self.height_scale = height_scale

        self.frame = None
        self.mask = None
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

        self.endpoint_rolling = np.zeros((endpoint_rolling_mean_range, 2), dtype=int)
        self.endpoint = None
        self.time_to_endpoint = None

        self.bot_radius = bot_radius

    def get_freq(self):
        return 1 / self.cam.get(cv.CAP_PROP_FPS)

    # get min and max y limits for bot
    def get_arena_constraints(self):
        max_y = (self.height - self.arena_height)/2 + self.bot_radius/2
        min_y = (self.height + self.arena_height)/2 - self.bot_radius/2

        max_bot_pos, _ = self._cam_to_bot(position = (self.endpoint_threshold, max_y))
        min_bot_pos, _ = self._cam_to_bot(position = (self.endpoint_threshold, min_y))

        return np.asarray((min_bot_pos, max_bot_pos))

    # convert from Camera Frame (pixels) to Bot Frame (meters)
    def _cam_to_bot(self, position: Tuple[int] = None, velocity: Tuple[int] = None):    
        if position is not None:
            pos_m = np.zeros(2, dtype = np.double) # m
            pos_m[0] = position[0] / self.px_per_meter_x + self.bot_offset
            pos_m[1] = - (position[1] - self.height_scale*self.height) / self.px_per_meter_y
        else:
            pos_m = None

        if velocity is not None:
            vel_m_s = np.zeros(2, dtype = np.double) # m/s
            vel_m_s[0] = velocity[0] / self.px_per_meter_x 
            vel_m_s[1] = -velocity[1] / self.px_per_meter_y
        else:
            vel_m_s = None

        return pos_m, vel_m_s
    
    # convert from Bot Frame (meters) to Camera Frame (pixels)
    def _bot_to_cam(self, position: NDArray[np.double] = None, velocity: NDArray[np.double] = None):
        if position is not None:
            pos_px = np.zeros(2, dtype = int) # m
            pos_px[0] = (position[0] - self.bot_offset) * self.px_per_meter_x
            pos_px[1] = -position[1] * self.px_per_meter_y + self.height_scale*self.height 
        else:
            pos_px = None

        if velocity is not None:
            vel_px_s = np.zeros(2, dtype = int) # m/s
            vel_px_s[0] = velocity[0] * self.px_per_meter_x 
            vel_px_s[1] = -velocity[1] * self.px_per_meter_y 
        else:
            vel_px_s = None

        return pos_px, vel_px_s
            
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
            self.mask = mask

            # Find largest contour: largest swath of white mask
            cntrs = cv.findContours(mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            cntrs = imutils.grab_contours(cntrs)
            center = None

            # Old pos for velocity calc, reset current
            last_pos = self.ball_pos
            self.ball_pos = None

            if len(cntrs) > 0:
                c = max(cntrs, key = cv.contourArea)
                ((x, y), radius) = cv.minEnclosingCircle(c)

                if radius > 20:
                    # Moments - unnecessary?
                    # M = cv.moments(c)
                    # center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                    
                    # Old pos for velocity calc
                    # last_pos = self.ball_pos

                    # Update ball position
                    self.ball_pos_rolling = rolling_add(self.ball_pos_rolling, np.array((int(x), int(y)), dtype=int))
                    # self.ball_pos = np.mean(self.ball_pos_rolling, axis=0, dtype=int)
                    self.ball_pos = np.asarray(rolling_average(self.ball_pos_rolling), dtype=int)

                    loop_time = time.perf_counter_ns()

                    # Calculate instantaneous ball velocity
                    if last_pos is not None:
                        dt = (loop_time - self.last_time) / (1e9) # convert to s
                        new_vel = np.array((self.ball_pos - last_pos) / dt)

                        self.ball_vel_rolling = rolling_add(self.ball_vel_rolling, new_vel)
                        self.ball_vel = np.mean(self.ball_vel_rolling, axis=0)
                        self.ball_vel = np.asarray(rolling_average(self.ball_vel_rolling))

                    
                    self.last_time = loop_time        

                    self.ball_radius_measured = int(radius)
                
        return ret

    def predict_path(self, dt:float = 0.5):
        '''
        Predicts path of ball until it hits a wall or reaches the endpoint threshold
        '''
        # Clear endpoint
        self.endpoint = None
        # Check if ball found
        if self.ball_pos is not None and self.ball_vel is not None:
            [bx, by] = self.ball_pos

            # Take rolling average
            [vx, vy] = self.ball_vel

            # Absolute y max given arena bounds
            y_max = self.arena_height/2
            y_mid = self.height/2

            # Reset ball path
            self.clear_predicted_path()

            # Ensure ball is coming towards robot with significant velocity
            if (vx < 0) and (vx**2 + vy**2 > 50**2):
                self.time_to_endpoint = 0
                in_bounds = True

                # Threshold for path
                while bx > self.endpoint_threshold and in_bounds:
                    # Calculate incoming ball position differentials
                    dx = vx*dt
                    dy = vy*dt

                    in_bounds = y_mid - y_max + self.ball_radius < by + dy <  y_mid + y_max - self.ball_radius
                    
                    # New pos
                    bx = int(bx + dx)
                    by = int(by + dy)

                    # Store prediction step
                    self.ball_predicted_path = np.append(self.ball_predicted_path, np.array([[bx, by, 0]]), axis=0)
                    self.time_to_endpoint += dt

                if len(self.ball_predicted_path) != 0:
                    self.endpoint_rolling = rolling_add(self.endpoint_rolling, self.ball_predicted_path[-1, 0:2])
                    self.endpoint = np.asarray(rolling_average(self.endpoint_rolling), dtype=int)



    def predict_path_perfect_collisions(self, dt:float = 0.5):
        '''
        Casts ray from ball position w/ velocity vector
        to determine where ball will cross into some radius around
        the robot's base frame
        
        Assumes frictionless rolling and momentum preserved on collision

        :param epsr: epsilon r, % of reach to watch for ball cross
        :param dt: delta time, time resolution
        '''
        # Check if ball found
        if self.ball_pos is not None and self.ball_vel is not None:
            [bx, by] = self.ball_pos
            ball_px_x = bx - self.endpoint_threshold

            # Take rolling average
            [vx, vy] = self.ball_vel

            # Radius to watch for cross
            # r_prep = espr*1

            # Absolute y max given arena bounds
            y_max = self.arena_height/2
            y_mid = self.height/2

            # Reset ball path
            self.clear_predicted_path()

            # Ensure ball is coming towards robot with significant velocity
            if (vx < 0) and (vx**2 + vy**2 > 50**2):
                self.time_to_endpoint = 0

                # Threshold for path
                while bx > self.endpoint_threshold:
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
                    self.time_to_endpoint += dt

                    if len(self.ball_predicted_path) != 0:
                        self.endpoint_rolling = rolling_add(self.endpoint_rolling, self.ball_predicted_path[-1, 0:2])
                        self.endpoint = np.mean(self.endpoint_rolling, axis=0)

    def generate_bot_goal(self):
        '''
        Provides ball info for external use as Tuple with following balues

        :returns cam_info[0], ball_pos: X,Y coords in meters: ball position
        :returns cam_info[1], ball_endpoint: X, Y coords in meters: where ball is predicted to be heading
        :returns cam_info[2], time_to_endpoint: time in seconds before ball reaches current endpoint
        :returns cam_info[3], ball_vell: Vx,Vy coords in meters/s: ball velocity
        '''
        
        ball_pos_m, ball_vel_ms = self._cam_to_bot(position=self.ball_pos, velocity=self.ball_vel)
        ball_endpoint, _ = self._cam_to_bot(position = self.endpoint)

        return (ball_pos_m, ball_endpoint, self.time_to_endpoint, ball_vel_ms)



    def playback(self, bot_pos: NDArray[np.double] = None, bot_pos_d: NDArray[np.double] = None, show_mask: bool = False):
        '''
        Optional playback to show ball tracking and path prediction
        '''
        # Draw arena
        cv.line(self.frame, (0, int(self.height/2 + self.arena_height/2)), (self.width, int(self.height/2 + self.arena_height/2)), 0, 4)
        cv.line(self.frame, (0, int(self.height/2 - self.arena_height/2)), (self.width, int(self.height/2 - self.arena_height/2)), 0, 4)
        cv.line(self.frame, (self.endpoint_threshold, 0), (self.endpoint_threshold, self.height), 0, 4)


        # Draw bot and desired
        if bot_pos is not None:
            bot_pos, _ = self._bot_to_cam(position = bot_pos)
            cv.circle(self.frame, bot_pos, self.bot_radius, (45, 62, 247), 2)

        if bot_pos_d is not None:
            bot_pos_d, _ = self._bot_to_cam(position = bot_pos_d)
            cv.circle(self.frame, bot_pos_d, self.bot_radius, (144, 255, 245), 1)

        if self.ball_pos is not None:
            # Draw circle about ball 
            cv.circle(self.frame, self.ball_pos, self.ball_radius_measured, self.color[1], 2)
            cv.circle(self.frame, self.ball_pos, 5, (0, 255, 255), -1)

            # Draw circle about smoothed endpoint
            if self.endpoint is not None:
                cv.circle(self.frame, self.endpoint, self.ball_radius_measured, self.color[1], 2)
                cv.circle(self.frame, self.endpoint, 5, (0, 0, 255), -1)

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
        if show_mask:
            cv.imshow("Mask", self.mask)

        # kill key q
        return not cv.waitKey(1) == ord('q')

