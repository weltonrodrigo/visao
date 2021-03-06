import cv2
import acapture
import sys
import numpy as np
import colorsys

sys.path.append('./python_glview') 

import pyglview
from OpenGL.GLUT import *
from logging import warning

bg = cv2.imread('bg.jpg', cv2.IMREAD_COLOR)
bg = cv2.resize(bg, dsize=(1280, 720), interpolation=cv2.INTER_CUBIC)

viewer = pyglview.Viewer()
cap = acapture.open(0) # Camera 0,  /dev/video0


got_sample = False
sample_range = None
sample_size = (10, 10)
threshold = 20

kernel = np.ones((10,10), np.uint8)

def restrict(color):
    # https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    # Max and min values of H, S, V in opencv
    min = (0, 0, 0)
    max = (179, 255, 255)

    return np.clip(color, min, max)

def color_range4(sample):
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    low = hsv.min(axis=(0,1))
    high= hsv.max(axis=(0,1))
    print('Low is: ', low)
    print('High is: ', high)

    low = restrict(hsv.mean(axis=(0,1)) - [20, 100, 100])
    high = restrict(high + [4, 100, 100])

    print('Low is: ', low)
    print('High is: ', high)

    sample_range = (low, high)

    got_sample = True
def color_range3(sample):
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    low = hsv.min(axis=(0,1))
    high= hsv.max(axis=(0,1))
    print('Low is: ', low)
    print('High is: ', high)

    low = restrict(low - [4, 100, 100])
    high = restrict(high + [4, 100, 100])

    print('Low is: ', low)
    print('High is: ', high)

    sample_range = (low, high)

    got_sample = True
def color_range2(sample):
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    mean = hsv.mean(axis=(0,1))
    std = hsv.std(axis=(0,1))
    deviation=6 * std

    low = restrict(hsv.min(axis=(0,1))-deviation)
    high = restrict(hsv.max(axis=(0,1))+deviation)

    print('Low is: ', low)
    print('High is: ', high)

    sample_range = (low, high)

    got_sample = True

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

            bg_masked = np.copy(bg)

            bg_masked[mask==0] = [0, 0, 0]

            frame[mask != 0] = [0, 0, 0]

            frame = frame + bg_masked

        viewer.set_image(frame)

def sample_cb(c):
    global got_sample
    color_range2(c)

viewer.set_sample_size(sample_size)
viewer.set_sample_cb(sample_cb)
viewer.set_loop(loop)
viewer.start()
