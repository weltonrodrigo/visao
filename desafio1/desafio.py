import cv2
import acapture
import sys
import numpy as np

sys.path.append('./python_glview') 
import pyglview


##
## C to close window
## 1 and 2 to increase range of colors
## q and w to reduce
## r to reset
## click to sample

## May be necessary to resize the BG image based on your

bg = cv2.imread('bg.jpg', cv2.IMREAD_COLOR)

cap = acapture.open(0) # Camera 0,  /dev/video0

got_sample = False
sample_range = None
sample_size = (10, 10)
threshold = 20
step = 1
resized_bg = None

kernel = np.ones((10,10), np.uint8)


def resize_bg(frame_size):
    '''Necessary to resize the BG image'''
    global bg, resized_bg
    if resized_bg is not None:
        return resized_bg
    else:
        print (frame_size)
        resized_bg = cv2.resize(bg, dsize=(frame_size[1], frame_size[0]), interpolation=cv2.INTER_CUBIC)
        return resize_bg


def change(k):
    '''Increase or decrease the range or masked colors'''
    global sample_range

    if not got_sample:
        return None

    low = sample_range[0]
    high = sample_range[1]
    
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
    '''Clip color value to a valid HSV value in OpenCV'''
    # https://docs.opencv.org/trunk/df/d9d/tutorial_py_colorspaces.html
    # Max and min values of H, S, V in opencv
    min = (0, 0, 0)
    max = (179, 255, 255)

    return np.clip(color, min, max)

def color_range5(sample):
    '''Approach #5: based on the mean of a sample, but ignoring value component'''
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    # Range. 500 in V component will clip to 0 and 255
    amount = [10, 100, 500]

    low = restrict(hsv.mean(axis=(0,1)) - amount)
    high = restrict(hsv.mean(axis=(0,1)) + amount)

    sample_range = (low, high)

    got_sample = True


def color_range4(sample):
    '''Approach #4: band pass filter based on mean'''
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
    '''Approach #3: extend the min and the max values with a fixed amount'''
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
    '''Approach #2: analise the sample to find distribution of values
    and choose a window based on it'''
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
    '''Approach #1: get min and max values from the sample, use it as the range'''
    global sample_range, got_sample

    # turn the bytearray into an (x, x, 3) np array (sample size with 3 colors)
    img = np.frombuffer(sample, dtype=np.uint8).reshape(*sample_size, 3)

    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    print('HSV is:', hsv)

    sample_range = (hsv.min(axis=(0,1)), hsv.max(axis=(0,1)))

    got_sample = True

def keyboard_listener(key, x, y):
    global got_sample

    print(chr(key))
    if key == ord('c'):
        exit(0)
    if key == ord('r'):
        got_sample=False
    if chr(key) in '123qwe':
        change(chr(key))

def sample_cb(c):
    '''Callback to get sample pixels from opengl window
    it come as a bytearray'''
    global got_sample
    color_range5(c)

def text(image):
    '''Add description text to the frame'''
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

            # make a copy of the resized BG to apply mask
            bg_masked = np.copy(resize_bg(image_copy.shape[:2]))

            bg_masked[mask==0] = [0, 0, 0]

            frame[mask != 0] = [0, 0, 0]

            frame = frame + bg_masked

        # Add text
        frame = text(frame)

        # Set the frame to OpenGl window
        viewer.set_image(frame)

viewer = pyglview.Viewer(keyboard_listener=keyboard_listener)
viewer.set_sample_size(sample_size)
viewer.set_sample_cb(sample_cb)
viewer.set_loop(loop)
viewer.start()
