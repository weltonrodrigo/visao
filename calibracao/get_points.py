import logging
import sys
from enum import Enum
from typing import Optional

import cv2
import numpy as np
from numpy.core._multiarray_umath import ndarray

import calibration
from calibration import CameraParams, Calibration

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ll = logging.getLogger(__file__)


class Command(Enum):
    COMMAND_RESTART = 1
    COMMAND_QUIT = 2
    COMMAND_CONTINUE = 3


class GetPoints():
    points: ndarray

    def __init__(self, video_source, title: str, params: Optional[CameraParams] = None, fps=30, rectify=True):
        self.source = video_source
        self.title = title
        self.wait = 1000 // fps
        self.source = self.source
        self.params = params
        self.points: np.ndarray = None
        self.target = 0
        self.rectify = rectify

    def _init_points(self) -> None:
        # TODO: init with invisible points (maybe out of frame)
        self.points = np.array([[[20, 20]], [[50, 20]], [[20, 50]], [[50, 50]]], np.float32)

    def draw_polygon(self, frame):
        # Draw a diagonal blue line with thickness of 5 px
        # cv2.line(frame, self.points[0], self.points[1], (255, 0, 0), 5)
        # cv2.line(frame, self.points[1], self.points[3], (255, 0, 0), 5)
        # cv2.line(frame, self.points[3], self.points[2], (255, 0, 0), 5)
        # cv2.line(frame, self.points[2], self.points[0], (255, 0, 0), 5)
        cv2.drawChessboardCorners(frame, (2, 2), self.points, False)

    def on_keyboard(self, key) -> bool:
        key = key & 0xFF
        if key == ord('q'):
            return Command.COMMAND_QUIT
        elif key == ord('r'):
            return Command.COMMAND_RESTART
        elif key == ord(' '):
            # Pause
            cv2.waitKey()
            return Command.COMMAND_CONTINUE

    def on_mouse(self, event, x, y, flags, userdata) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.target > len(self.points) - 1:  # passed last points, reinit.
                self.target = 0
            self.points[self.target, :] = (x, y)
            self.target += 1
            ll.info((event, x, y, flags, userdata))

    def initUndistort(self):
        self.params.newcameramtx, self.params.roi = (
            Calibration.get_optimal_camera_matrix(self.params)
        )

        self.mapx, self.mapy = Calibration.get_undistort_maps(self.params)


    def _setup(self):
        self._init_points()
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap:
            ll.error(f'Could not open video source {self.source}')
        if self.rectify:
            self.initUndistort()
        cv2.namedWindow(self.title)
        cv2.setMouseCallback(self.title, self.on_mouse)

    def _loop(self) -> Command:
        while self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                break

            if self.rectify:
                frame = cv2.remap(frame,self.mapx, self.mapy, interpolation=cv2.INTER_CUBIC)

            self.draw_polygon(frame)
            cv2.imshow(self.title, frame)

            # ~ 30 frames per second
            key = cv2.waitKey(self.wait)

            if key != -1:
                command = self.on_keyboard(key)
                if command != Command.COMMAND_CONTINUE:
                    return command

    def run(self) -> np.ndarray:
        self._setup()
        command = self._loop()
        self._stop()
        if command == Command.COMMAND_QUIT:
            self._quit()
        elif command == Command.COMMAND_RESTART:
            self.restart()
        return self.points

    def restart(self):
        self.run()

    def _stop(self):
        cv2.destroyAllWindows()
        self.cap.release()

    def _quit(self):
        ll.info(f'Points are: {self.points}')
        return self.points


if __name__ == '__main__':
    params = CameraParams.read_from_file('calib_files/boofcv-camera-1.yaml')
    app = GetPoints(
        video_source='trabalho1/camera1.webm',
        title='Camera 1',
        fps=30,
        params=params,
        rectify=True
    )
    app.run()
