import cv2
import acapture
import sys
import numpy as np
import colorsys

sys.path.append('./python_glview') 

import pyglview
from OpenGL.GLUT import *
from logging import warning

viewer = pyglview.Viewer()
cap = acapture.open(0) # Camera 0,  /dev/video0


got_sample = False
sample_range = None
sample_size = (5, 5)
threshold = 20

kernel = np.ones((10,10), np.uint8)

def color_range(sample):
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    sample_range = (hsv.min(axis=(0,1)), hsv.max(axis=(0,1)))

    got_sample = True

def loop():
    check,frame = cap.read() # non-blocking

    if check:
        if got_sample:
            image_copy = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

            mask = cv2.inRange(image_copy, *sample_range)

            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


            frame[mask != 0] = [0, 255, 0]

        viewer.set_image(frame)

def sample_cb(c):
    global got_sample
    color_range(c)

viewer.set_sample_size(sample_size)
viewer.set_sample_cb(sample_cb)
viewer.set_loop(loop)
viewer.start()
