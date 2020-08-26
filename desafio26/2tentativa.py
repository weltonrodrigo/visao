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

cap = acapture.open(0) # Camera 0,  /dev/video0


got_sample = False
sample_range = None
sample_size = (10, 10)
threshold = 20
step = 1

kernel = np.ones((10,10), np.uint8)

def change(k):
    global sample_range

    low = sample_range[0]
    high = sample_range[1]
    
    print('k is', k)

    if k == '1':
        low[0] = low[0] - step
        high[0] = high[0] + step
    if k == '2':
        low[1] = low[1] - step
        high[1] = high[1] + step
    if k == '3':
        low[2] = low[2] - step
        high[2] = high[2] + step

    if k == 'q':
        low[0] = low[0] + step
        high[0] = high[0] - step
    if k == 'w':
        low[1] = low[1] + step
        high[1] = high[1] - step
    if k == 'e':
        low[2] = low[2] + step
        high[2] = high[2] - step

    sample_range = (restrict(low), restrict(high))
    
    print('new range is', sample_range)

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

    amount = [10, 100, 100]

    low = restrict(hsv.mean(axis=(0,1)) - amount)
    high = restrict(hsv.mean(axis=(0,1)) + amount)

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

def keyboard_listener(key, x, y):
    print(key)
    print(chr(key))
    global got_sample
    if key == ord('c'):
        exit(0)
    if key == ord('r'):
        got_sample=False
    if chr(key) in '123qwe':
        change(chr(key))

def text(image):
    if got_sample:
        low=  'Low:  hsv({:3.0f}, {:3.0f}, {:3.0f})'.format(*sample_range[0])
        high= 'High: hsv({:3.0f}, {:3.0f}, {:3.0f})'.format(*sample_range[1])
    else:
        low = high = 'None'

    image = cv2.putText(image, low, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA)

    image = cv2.putText(image, high, (20, 65), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA)
    return image

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

        frame = text(frame)
        viewer.set_image(frame)

def sample_cb(c):
    global got_sample
    color_range4(c)

viewer = pyglview.Viewer(keyboard_listener=keyboard_listener)
viewer.set_sample_size(sample_size)
viewer.set_sample_cb(sample_cb)
viewer.set_loop(loop)
viewer.start()
