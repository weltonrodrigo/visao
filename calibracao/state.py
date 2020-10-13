from abc import ABC, abstractmethod
import calib2
import cv2
import logging
import numpy as np
import time

ll = logging.getLogger(__name__)


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
    def calibrator(self) -> calib2.Calibrator:
        return self._calibrator

    @calibrator.setter
    def calibrator(self, calibrator: calib2.Calibrator):
        self._calibrator = calibrator
        self.setup()

    @property
    def context_data(self) -> {}:
        return self._context_data

    @context_data.setter
    def context_data(self, context: {}):
        self._context_data = context

    def setup(self):
        self.status_msg = str(self)
        self.calibrator.set_loop(self.loop)
        self.calibrator.set_keyboard_listener(self.on_keyboard)

    @abstractmethod
    def loop(self):
        pass

    def paint(self, frame):
        ll.debug('will paint in %s state' % self)
        cv2.putText(
            frame, self._status_msg, (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
            1, (255, 0, 0), 2, cv2.LINE_AA)
        # ll.debug('done painting in %s state' % self)
        # ll.debug('frame is %s' % frame)
        return True, frame

    # @abstractmethod
    def on_keyboard(self, key):
        pass

    # @abstractmethod
    def on_mouse(self, event, x, y, flags, userdata):
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

        if check:
            ret, frame = self.paint(frame)
            # ll.debug('frame is %s' % frame)
            self.calibrator.set_image(frame)

    def on_keyboard(self, key):
        ll.info('Got %s key' % chr(key))
        if chr(key) == 's':
            self.calibrator.transition_to(StateSampling())
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
        'newcameramtx': None,
        'mapx': None,
        'mapy': None
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
            mtx, dist, (w, h), self.calibrator.config['alpha'], (w, h))

        mapx, mapy = cv2.initUndistortRectifyMap(
            mtx, dist, None, newcameramtx, (w, h), cv2.CV_32FC1)

        ll.info('Going to calculate')

        tot_error = 0
        for i in range(len(self._objpoints)):
            imgpoints2, _ = cv2.projectPoints(
                self._objpoints[i], rvecs[i], tvecs[i], mtx, dist)

            error = cv2.norm(
                self._imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            tot_error += error

        ll.info("Mean error: %f " % (tot_error / len(self._objpoints)))

        if ret:
            self._camera_params = {
                'roi': roi,
                'mtx': mtx,
                'dist': dist,
                'rvecs': rvecs,
                'tvecs': tvecs,
                'newcameramtx': newcameramtx,
                'mapx': mapx,
                'mapy': mapy
            }
            return True

    def done(self, frame):
        if self.calculate(frame):
            context = {'camera_params': self._camera_params}
            self.calibrator.write_params(self._camera_params)
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
            gray, (self._cbcol, self._cbrow),
            cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        # If found, add object points, image points (after refining them)
        if ret:
            self._objpoints.append(self._objp)

            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._criteria)

            # ll.info(corners2)

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
                frame, (self._cbcol, self._cbrow), self._imgpoints[-1],
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

    def on_keyboard(self, key):
        if chr(key) == 'r':
            self.reset_sampling()
        if chr(key) == 'a':
            ll.info('Aborted sampling.')
            self.calibrator.transition_to(StateInitial())

        ll.info('Got %s key' % chr(key))
        ll.info('Will do nothing')


class StateCalibrated(State):
    _name = 'CALIBRATED'

    _show_original = False
    _crop = False

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
        p = self.context_data['camera_params']

        if not self._show_original:
            frame = cv2.remap(frame, p['mapx'], p['mapy'], cv2.INTER_LINEAR)

            if self._crop:
                # crop the result
                x, y, w, h = p['roi']
                frame = frame[y:y + h, x:x + w]

        ret, frame = super().paint(frame)
        return ret, frame

    def on_keyboard(self, key):
        if key == ord('r'):
            self.calibrator.transition_to(StateInitial())
        if key == ord('o'):
            self.show_original(False if self._show_original else True)
        if key == ord('c'):
            self._crop = False if self._crop else True

