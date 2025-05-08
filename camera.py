import cv2 as cv
import imutils

class cameraModule:
    """
    Class for handling camera and cv calls
    """
    def __init__(self, cptre, color_bounds):
        """
        :param cptre: opencv capture port, usually 0
        """
        self.cam = cv.VideoCapture(cptre)
        self.color = color_bounds

        

    def find_ball(self, debug = False):
        """
        Grabs camera frame and does computer vision
            :param debug: boolean, whether to show playback
            :return: success: 
            :return coords: either None or 2x1 x,y in pixels (?)
        """
        ret, frame = self.cam.read()

        if ret:
            # Perform CV
            # resize, blur, convert to HSV
            frame = imutils.resize(frame, width=600)
            blurred = cv.GaussianBlur(frame, (11, 11), 0)
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
                ((x,y), radius) = cv.minEnclosingCircle(c)

                if radius > 10:
                    M = cv.moments(c)
                    center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

            if debug:
                # Draw tracking marker if circle found
                if center:
                    cv.circle(frame, (int(x), int(y)), int(radius), self.color[1], 2)
                    cv.circle(frame, center, 5, (0, 0, 255), -1)

                # Show playback
                cv.imshow("Camera", frame)
                cv.imshow("Mask", mask)

                coords = None

                # Kill key q
                if cv.waitKey(1) == ord('q'):
                    return False, coords
            
            return True, coords
                
        return False, None

