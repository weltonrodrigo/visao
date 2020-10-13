import logging
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

import cv2
import numpy as np

ll = logging.getLogger(__name__)


def flags2names(flags: int):
    """Return names of cv2.CALIB_… flags activated
    based on an int number"""

    all_vars = vars(cv2)
    calib_vars = dict(filter(
        lambda tup: 'CALIB_' in tup[0], all_vars.items()))

    name_actives = map(lambda tup: tup[0], filter(
        lambda tup: flags & tup[1], calib_vars.items()))

    return "|".join(name_actives)


@dataclass
class CameraParams:
    camera_matrix: np.ndarray
    distortion_coefficients: np.ndarray
    image_height: int
    image_width: int
    total_error: int = field(default=-1)
    roi: Tuple[int, int, int, int] = (0, 0, 0, 0)
    rvecs: List = field(default_factory=list)
    tvecs: List = field(default_factory=list)
    newcameramtx: np.ndarray = np.array([])
    mapx: np.ndarray = np.array([])
    mapy: np.ndarray = np.array([])
    per_view_error: List = field(default_factory=list)

    def write_to_file(self, filename: str, comment="") -> None:
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_WRITE)
        try:
            fs.writeComment(comment)
            for name, value in self.__dict__.items():
                if type(value) == list:
                    fs.startWriteStruct(name, flags=cv2.FILE_NODE_SEQ)
                    for item in value:
                        fs.write('', item)
                    fs.endWriteStruct()
                else:
                    fs.write(name, value)
            fs.release()
        except Exception as e:
            ll.exception(f'Error writing params to file {filename}.')

    @staticmethod
    def read_from_file(filename: str) -> 'CameraParams':
        fs = cv2.FileStorage(filename, cv2.FILE_STORAGE_READ)
        if not fs.isOpened():
            raise (FileNotFoundError(filename))
        params = {}
        for name in fs.root().keys():
            node = fs.getNode(name)
            if node.isMap():  # ndarrays are maps
                params[name] = node.mat()
            elif node.isSeq():  # lists are seqs
                params[name] = [
                    node.at(index).mat() for index in range(node.size())
                ]
            elif node.isInt():
                params[name] = int(node.real())
            elif node.isReal():
                params[name] = node.real()
            else:
                ll.error(f'Type {node.type()} of node {name} not supported yet. Continuing.')
        fs.release()
        return CameraParams(**params)


class ChessBoardCalibration:
    """"Detection of a calibration object in the form of a chessboard"""
    _chess_flags = (
            cv2.CALIB_CB_FAST_CHECK
            | cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
    )

    _criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001
    )

    def __init__(self, cbrow: int, cbcol: int, size: int = 1):
        self._cbrow = cbrow
        self._cbcol = cbcol

        # Generate points with size 1
        self._objp = np.zeros((self._cbrow * self._cbcol, 3), np.float32)
        self._objp[:, :2] = (
            np.mgrid[0:self._cbcol, 0:self._cbrow].T.reshape(-1, 2)
        )

        # use actual dimensions of rectangles
        self._objp * [size, size, 0]

        self._objpoints = []
        self._imgpoints = []

    def find(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        success, corners = cv2.findChessboardCorners(
            gray, (self._cbcol, self._cbrow), self._chess_flags)

        # Image do contain a chessboard, so a sample
        # will be created from it
        if success:
            # Do a finer pass looking for chessboard corners
            corners2 = cv2.cornerSubPix(
                gray, corners, (11, 11), (-1, -1), self._criteria)

            # ll.debug(f'Detected corners:\n{np.array2string(corners, separator=",")}')
            # ll.debug(f'Refined corners:\n{np.array2string(corners2, separator=",")}')

            return self._objp, corners2
        else:
            return None

    def paint(self, img, corners):
        cv2.drawChessboardCorners(
            img, (self._cbcol, self._cbrow), corners, True)


class Calibration:
    _calib_flags = 0

    _criteria = \
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self, delay: int, sample_size: int, cbcols, cbrows, alpha=0, nooutliers=True):
        self._delay = delay
        self._alpha = alpha
        self._sample_size = sample_size
        self._samples = {'imgpoints': [], 'objpoints': [], 'comments': []}
        self._sample_img_dimensions = (0, 0)
        self._cbrow = cbrows
        self._cbcol = cbcols
        self._nooutliers = nooutliers
        self._board = \
            ChessBoardCalibration(self._cbrow, self._cbcol)

    def add_sample(self, img: [], comment: str = "") -> int:
        """Add an image to the sample"""
        points = self._board.find(img)
        if points:
            worldpoints, imgpoints = points
            self._samples['objpoints'].append(worldpoints)
            self._samples['imgpoints'].append(imgpoints)
            self._samples['comments'].append(comment)

            # if this is the first image, use its dimensions
            # (all images must be the same dimension
            count = len(self._samples['objpoints'])
            if count == 1:
                self._sample_img_dimensions = img.shape[:-1]
            else:
                if img.shape[:-1] != self._sample_img_dimensions:
                    raise ValueError(
                        f'All images must be the same dimension. '
                        f'First sample: {self._sample_img_dimensions} '
                        f'This sample: {img.shape[:-1]}'
                    )
            return count

    def has_outliers(self, errors: np.array):
        """"Check for values bigger than 2 standard deviations in the list"""
        # Errors has shape (N, 1)
        errors = errors.flatten()
        z_score = (errors - errors.mean()) / errors.std()
        tup = np.where(z_score > 2)

        # npwhere returns a single element tuple with a 1 dimension array.
        return tup[0].tolist()

    def purge_outliers(self, indexes: List[int]):
        # Remove the sample from the lists.
        for key in self._samples.keys():
            # must be in reverse so the removal change array indexes
            for i in sorted(indexes, reverse=True):
                del self._samples[key][i]

    def calculate_camera_params(self, flags=None) -> CameraParams:
        """ Return camera intrinsic matrix based on sample images"""
        h, w = self._sample_img_dimensions
        flags = flags if flags else self._calib_flags

        ll.info('Going to calculate camera params')
        ll.info(f'Using flags {flags2names(flags)}')

        dist, mtx, per_view_errors, rvecs, total_error, tvecs = (
            self.do_calibrate(flags, h, self._samples['imgpoints'], w, self._samples['objpoints'])
        )

        if self._nooutliers:
            indexes = self.has_outliers(per_view_errors)
            if indexes:
                ll.info('Will purge outliers')
                msgs = [self._samples['comments'][i] for i in indexes]
                errors = [per_view_errors[i] for i in indexes]
                self.print_error_stats(msgs, errors, total_error)

                self.purge_outliers(indexes)
                return self.calculate_camera_params(flags=flags)

        self.print_error_stats(self._samples['comments'], per_view_errors, total_error)

        if total_error:  # will be Falsy value if calibrate doesn't work.
            return CameraParams(
                camera_matrix=mtx,
                distortion_coefficients=dist,
                image_height=self._sample_img_dimensions[0],
                image_width=self._sample_img_dimensions[1],
                # roi=roi,
                rvecs=rvecs,
                tvecs=tvecs,
                total_error=total_error,
                per_view_error=per_view_errors
                # newcameramtx=newcameramtx,
                # mapx=mapx,
                # mapy=mapy
            )

    def do_calibrate(self, flags, heigth, img_points, width, obj_points):
        """"Do calibration using extended version of calibrateCameraRO
        see: https://docs.opencv.org/4.4.0/d9/d0c/group__calib3d.html#ga11eeb16e5a458e1ed382fb27f585b753
        """
        # fixed point will be upper right
        fixed_point = self._cbcol - 1

        (total_error, mtx, dist, rvecs, tvecs, newObjPts,
         stdDeviationsIntrinsics, stdDeviationsExtrinsics,
         stdDeviationsObjPoints, perViewErrors
         ) = cv2.calibrateCameraROExtended(
            objectPoints=obj_points, imagePoints=img_points,
            imageSize=(width, heigth), cameraMatrix=None, distCoeffs=None,
            iFixedPoint=fixed_point, flags=flags, criteria=self._criteria)
        return dist, mtx, perViewErrors, rvecs, total_error, tvecs

    def print_error_stats(self, msgs, errors, total_error):
        ll.info(f'REPROJECTION RMS ERROR IS {total_error}')
        ll.info('PER VIEW ERRORS:')

        combined = list(zip(errors, msgs))

        # sort by descending error size
        combined.sort(key=lambda tup: tup[0], reverse=True)

        for error, msg in combined:
            ll.info(f'{error}\t{msg}')

    @staticmethod
    def get_optimal_camera_matrix(params: CameraParams, new_image_size: Optional[Tuple[int, int]] = None,
                                  alpha: float = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Get new optional camera matrix for the provided distortion coefficients
        The new camera matrix considers the scaling needed to crop the image and avoid
        black areas
        :returns (new_camera_matrix, region-of-interest)
        """
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=params.camera_matrix,
            distCoeffs=params.distortion_coefficients,
            imageSize=(params.image_width, params.image_height),
            newImgSize=None, alpha=alpha
        )

        return newcameramtx, roi


    @staticmethod
    def get_undistort_maps(params: CameraParams, R = None) -> Tuple[np.ndarray, np.ndarray]:
        mapx, mapy = cv2.initUndistortRectifyMap(
            cameraMatrix=params.camera_matrix,
            distCoeffs=params.distortion_coefficients,
            R=R,
            newCameraMatrix=params.newcameramtx,
            size=(params.image_width, params.image_height),
            m1type=cv2.CV_32FC2
        )
        return mapx, mapy
