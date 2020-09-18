import cv2
import acapture
import numpy as np
from argparse import ArgumentParser
import sys
import time
import logging
from abc import ABC, abstractmethod

from imutils.video import FPS
import imutils
from imutils.video import FileVideoStream

sys.path.append('../desafio26/python_glview')
import pyglview

# # Config
# cbrow = 10
# cbcol = 7
# calibration_sample_size = 10

# # State
# State = 'INITIAL'
# calibrating = False
# calculate_calibration = False
# calibrated = False
# camera_params = (None, None, None)
# collected_samples_size = 0

# # Holds the video reading object
# cap = None

# # termination criteria
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((cbrow * cbcol, 3), np.float32)
# objp[:, :2] = np.mgrid[0:cbcol, 0:cbrow].T.reshape(-1, 2)

# # Arrays to store object points and image points from all the images.
# objpoints = []  # 3d point in real world space
# imgpoints = []  # 2d points in image plane.


# def reset_calibration():
#     global objpoints, imgpoints, calibrated, camera_params
#     objpoints = []
#     imgpoints = []
#     camera_params = (None, None, None)
#     calibrated = False


# def capture(frame):
#     global collected_samples_size, calibrating, calculate_calibration


#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#   # Find the chess board corners
#     ret, corners = cv2.findChessboardCorners(gray, (cbrow, cbcol), cv2.CALIB_CB_FAST_CHECK)

#   # If found, add object points, image points (after refining them)
#     if ret:
#         objpoints.append(objp)

#         corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
#         imgpoints.append(corners2)

#         # Draw and display the corners
#         #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame = cv2.drawChessboardCorners(frame, (cbrow, cbcol), corners2,ret)

#         collected_samples_size += 1
#         print("Collected %s samples" % collected_samples_size)

#         if collected_samples_size >= calibration_sample_size:
#             calibrating, calculate_calibration = False, True
#             collected_samples_size = 0

#         return (True, frame)

#           else:
#               return (False, frame)



# def loop():
#     global camera_params, calibrated, calculate_calibration

#     check, frame = cap.read()

#     if check:
#       if calibrating:
#           ret, image = capture(frame)
#           if ret:
#             frame = image
#       if calculate_calibration:
#         print('Done calculating, now calibration')

#         #print('objpoints is', objpoints)
#         #print('imgpoints is', imgpoints)
#         h, w, _ = frame.shape
#         ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, (h, w),None,None)
#         #print(ret, mtx, dist, rvecs, tvecs)

#         if ret:
#           newcameramtx, roi =cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
#           camera_params = (mtx, newcameramtx, dist)
#           calibrated = True
#           calculate_calibration = False
#           print("Camera matrix : \n")
#           print(mtx)
#           print("dist : \n")
#           print(dist)
#           print("rvecs : \n")
#           print(rvecs)
#           print("tvecs : \n")
#           print(tvecs)

#       if calibrated:
#         frame  = cv2.undistort(frame, camera_params[0], camera_params[2], None, camera_params[1])

#       viewer.set_image(frame)
#     else:
#         print("Frame not read")

# def on_keyboard(key, x, y):
#   global calibrating
#   reset_calibration()
#   calibrating = True

#   return True

class Calibrator(ABC):

    _state = None
    _cap = None
    _viewer = None
    _config = {
        'cbrow': 7,
        'cbcol': 10,
        'samples': 30,
        'delay': 1
    }

    @property
    def config(self):
        return self._config

    def transition_to(self, state, context={}):
        ll.info('Entering %s state' % state)
        self._state = state
        state.context = context
        state.calibrator = self

    def read(self):
        return self._cap.read()

    def set_image(self, image):
        return self._viewer.set_image(image)

    def __init__(self):
        self._viewer = pyglview.Viewer()

    def set_loop(self, loop):
        self._viewer.set_loop(loop)

    def set_keyboard_listener(self, func):
        self._viewer.keyboard_listener = func

    def start(self, device):
        self._cap = acapture.open(device)
        self.transition_to(StateInitial(), context={})
        self._viewer.start()


class State(ABC):
    """Base classe for state behaviour"""

    _status_msg = None

    @property
    def status_msg(self) -> str:
        return self._status_msg

    @status_msg.setter
    def status_msg(self, msg: str) -> str:
        self._status_msg = msg
        return msg

    @property
    def calibrator(self) -> Calibrator:
        return self._calibrator

    @calibrator.setter
    def calibrator(self, calibrator: Calibrator):
        self._calibrator = calibrator
        self.setup()

    @property
    def context(self) -> {}:
        return self._context

    @context.setter
    def context(self, context: {}):
        self._context = context

    def setup(self):
        self.status_msg = str(self)
        self.calibrator.set_loop(self.loop)
        self.calibrator.set_keyboard_listener(self.on_keyboard)

    @abstractmethod
    def loop():
        pass

    def paint(self, frame):
        ll.debug('will paint in %s state' % self)
        cv2.putText(
            frame, self._status_msg, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA)
        # ll.debug('done painting in %s state' % self)
        # ll.debug('frame is %s' % frame)
        return True, frame

    @abstractmethod
    def on_keyboard(self, key, x, y):
        pass

    def __str__(self):
        return self._name


class StateInitial(State):
    _name = 'INITIAL'

    _loglevel = logging.INFO

    def setup(self):
        super().setup()
        ll.setLevel(logging.INFO)

    def loop(self):
        check, frame = self.calibrator.read()
        ll.info('shape is:', frame.shape)

        if check:
            ret, frame = self.paint(frame)
            # ll.debug('frame is %s' % frame)
            self.calibrator.set_image(frame)

    def on_keyboard(self, key, x, y):
        ll.info('Got %s key' % chr(key))
        if chr(key) == 's':
            self.calibrator.transition_to(StateSampling(), context={})
        if chr(key) == 'd':
            ll.setLevel(logging.DEBUG)

class StateSampling(State):
    _name = 'SAMPLING'

    _cbrow = None
    _cbcol = None
    _max_samples = None
    _got_sample = None
    _last_sample = time.time()
    _delay = None

    # Object holding an array representing the points in the checkboard
    _objp = None

    _criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    _imgpoints = []
    _objpoints = []

    _camera_params = {
        'ret': None,
        'mtx': None,
        'dist': None,
        'rvecs': None,
        'tvecs': None,
        'newcameramtx': None
    }

    _samples = 0

    def setup(self):
        self._cbrow = self.calibrator.config['cbrow']
        self._cbcol = self.calibrator.config['cbcol']
        self._max_samples = self.calibrator.config['samples']
        self._delay = self.calibrator.config['delay']

        # Initialize object representing point of checkboard
        self._objp = np.zeros((self._cbrow * self._cbcol, 3), np.float32)
        self._objp[:, :2] = np.mgrid[0:self._cbcol, 0:self._cbrow].T.reshape(
            -1, 2)

        # called last because setup uses the new loop and it must be ready
        # to capture
        super().setup()

        self.update_msg()

        ll.info('State %s is setup()' % self)

    def calculate(self, frame):
        h, w, _ = frame.shape
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            self._objpoints, self._imgpoints, (w, h), None, None)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, dist, (w, h), 1, (w, h))
        # TODO: invert dimensions of the image
        if ret:
            self._camera_params = {
                'ret': ret,
                'mtx': mtx,
                'dist': dist,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'newcameramtx': newcameramtx
            }
            return True

    def done(self, frame):
        if self.calculate(frame):
            context = {'camera_params': self._camera_params}
            self.calibrator.transition_to(StateCalibrated(), context)
        else:
            ll.error('Error calculating calibration, will reset.')
            self.reset_sampling()


    def update_msg(self):
        self.status_msg = 'Sampling: %s/%s' % (
            self._samples, self._max_samples)

    def sample(self, frame):

        if time.time() - self._last_sample < self._delay:
            self._got_sample = False
            return False

        if self._samples == self._max_samples:
            self._got_sample = False
            self.done(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(
            gray, (self._cbrow, self._cbcol),
            cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
            self._objpoints.append(self._objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._criteria)

            self._imgpoints.append(corners2)

            self._last_sample = time.time()

            ll.info('Got sample %s/%s' % (self._samples, self._max_samples))
            self._samples += 1

            self.update_msg()

            self._got_sample = True
            return True
        else:
            ll.info('Got NO sample')
            self._got_sample = False
            return False

    def paint(self, frame):
        ret, frame = super().paint(frame)

        if self._samples > 0:
            cv2.drawChessboardCorners(
                frame, (self._cbrow, self._cbcol), self._imgpoints[-1],
                self._got_sample)
        return ret, frame

    def loop(self):
        check, frame = self.calibrator.read()
        ll.debug('Got frame in %s State and check is %s' % (self, check))
        if check:
            self.sample(frame)
            ret, painted = self.paint(frame)

            self.calibrator.set_image(painted if ret else frame)

    def reset_sampling(self):
        self.calibrator.transition_to(StateSampling())

    def on_keyboard(self, key, x, y):
        if key == chr('r'):
            self.reset_sampling()

        ll.info('Got %s key' % chr(key))
        ll.info('Will do nothing')


class StateCalibrated(State):
    _name = 'CALIBRATED'

    _show_original = False

    def loop(self):
        check, frame = self.calibrator.read()
        if check:
            ret, painted = self.paint(frame)
            if ret:
                self.calibrator.set_image(painted)
                return

        self.calibrator.set_image(frame)
        return

    def setup(self):
        super().setup()

    def show_original(self, flag: bool):
        self._show_original = flag
        self._status_msg = "RAW" if flag else "CALIBRATED"

    def paint(self, frame):
        p = self.context['camera_params']

        if not self._show_original:
            frame = cv2.undistort(
                frame, p['mtx'], p['dist'], None, p['newcameramtx'])
        ret, frame = super().paint(frame)
        return ret, frame

    def on_keyboard(self, key, x, y):
        if key == ord('r'):
            self.calibrator.transition_to(StateInitial(), {})
        if key == ord('o'):
            self.show_original(False if self._show_original else True)


if __name__ == '__main__':
    ll = logging.getLogger(__name__)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(
        logging.Formatter("%(asctime)s [%(filename)s:%(lineno)d] %(message)s"))
    ll.addHandler(handler)
    ll.setLevel(logging.DEBUG)

    calib = Calibrator()
    calib.start(0)
