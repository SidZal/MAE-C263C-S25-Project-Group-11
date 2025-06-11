# get modules from project file
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from camera import cameraModule

# Color bounds
pink = [(150, 50, 20), (170, 255, 255)]
purple = [(150, 80, 80), (180, 255, 255)]
red = [] # Red wraps around Hue value, and that makes it harder </3

cam = cameraModule(4, purple, 400, 10, endpoint_threshold=200)

while True:
    if cam.find_ball():
        cam.predict_path(dt=0.1)
        if not cam.playback(show_mask=True):
            break