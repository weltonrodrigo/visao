import argparse
import itertools
import logging
import string
import sys
from pathlib import Path
from pprint import pformat
from typing import Tuple, List

import cv2
import numpy as np
from matplotlib import pyplot

import utils
from calibration import CameraParams, Calibration
from get_points import GetPoints

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
ll = logging.getLogger(__file__)
logging.getLogger().setLevel(logging.DEBUG)

CACHE_DIR = Path('cache')

# <editor-fold desc="Argparser">
parser = argparse.ArgumentParser(
    prog="Stereo Rectify",
    description="""Do a stereo rectification based on camera intrinsic and 4\
                    points visible in both cameras."""
)
parser.add_argument(
    "-r",
    "--read-points-from-file",
    metavar="numpy_txt_file"
)
parser.add_argument(
    "-w",
    "--write-points-to-file",
    metavar="numpy_txt_file"
)

args = parser.parse_args()
# </editor-fold>

img_points: List[np.ndarray] = []

# Camera2.webm is left, camera1.webm is Right
sources = ('videos/cam-2-fix.mp4', 'videos/cam-1-fix.mp4')
param_files = ("calib_files/cam2-fix.yaml", "calib_files/cam1-fix.yaml")


def mark_points(video_filenames, param_filenames):
    for source, param_file in zip(video_filenames, param_filenames):
        params = CameraParams.read_from_file(param_file)
        app = GetPoints(
            video_source=source,
            title=source,
            fps=30,
            params=params,
            rectify=False
        )
        img_points.append(app.run())
    return img_points


if args.read_points_from_file:
    img_points = np.loadtxt(args.read_points_from_file, dtype=np.float32).reshape((2, 4, 1, 2))
else:
    img_points = mark_points(sources, param_files)
    if args.write_points_to_file:
        np.savetxt(args.write_points_to_file, np.array(img_points).reshape(2, -1), header="Shape(2,4,1,2)")

obj_points = [
    [[0, 0, 0]],
    [[1.4, 0, 0]],
    [[0, 2.6, 0]],
    [[1.4, 2.6, 0]],
]

obj_points = np.array(obj_points, np.float32)

img_points_L, img_points_R = img_points[0], img_points[1]

ll.debug(img_points_L)
ll.debug(img_points_R)


def update_camera_matrix(params: CameraParams) -> CameraParams:
    """"Treat the source as already undistorted. Thus, the new camera matrix becomes the matrix and the distortion
    coefficients are zeroed."""
    if params.newcameramtx is None:
        params.newcameramtx, _ = Calibration.get_optimal_camera_matrix(params)
    params.camera_matrix = params.newcameramtx
    params.newcameramtx = None
    params.distortion_coefficients = None
    return params


def solve_for_cam(opoints, ipoints, params: CameraParams) -> Tuple[np.ndarray, np.ndarray]:
    """If use_new_mtx is True, consider the video already undistorted and scaled to region of interest.
    Set the new matrix as the camera matrix and set params accordingly.
    """
    _, rvec, tvec = cv2.solvePnP(
        objectPoints=opoints, imagePoints=ipoints,
        cameraMatrix=params.camera_matrix,
        distCoeffs=params.distortion_coefficients,
        flags=cv2.SOLVEPNP_P3P
    )
    return rvec, tvec


# Read params and use new camera matrix and no distortion,
# as source has already been undistorted.
params_L, params_R = map(CameraParams.read_from_file, param_files)
params_L, params_R = map(update_camera_matrix, (params_L, params_R))

rvec_L, tvec_L = solve_for_cam(opoints=obj_points, ipoints=img_points_L, params=params_L)
rvec_R, tvec_R = solve_for_cam(opoints=obj_points, ipoints=img_points_R, params=params_R)

ll.debug('Left translation vector: %s', pformat(tvec_L))
ll.debug('Right translation vector: %s', pformat(tvec_R))

(error, cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, R, T, E, F) = (
    cv2.stereoCalibrate(
        objectPoints=obj_points[None, ...],  # Add a list dimension
        imagePoints1=[img_points_L], imagePoints2=[img_points_R],
        cameraMatrix1=params_L.camera_matrix, distCoeffs1=params_L.distortion_coefficients,  # NOQA
        cameraMatrix2=params_R.camera_matrix, distCoeffs2=params_R.distortion_coefficients,  # NOQA
        imageSize=(1280, 720), flags=cv2.CALIB_FIX_INTRINSIC
    ))

ll.debug('Stereo Calibrate error: %s', error)
SCALE = 1
R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
    cameraMatrix1=params_L.camera_matrix,
    distCoeffs1=params_L.distortion_coefficients,
    cameraMatrix2=params_R.camera_matrix,
    distCoeffs2=params_R.distortion_coefficients,
    imageSize=(1280, 720),
    newImageSize=(int(1280*SCALE), int(720*SCALE)),
    R=R,
    T=T,
    alpha=1
)
ll.debug('T is \n%s', T)
ll.debug('CAm MTX L: \n%s', repr(params_L.camera_matrix))

rotations = np.linspace(np.identity(3), R1, num=10)
matrices = np.linspace(np.hstack((params_L.camera_matrix, np.zeros((3, 1)))), P1, num=10)


def get_map(args):
    matrix, rotation = args
    return cv2.initUndistortRectifyMap(
        cameraMatrix=params_L.camera_matrix,
        distCoeffs=params_L.distortion_coefficients,
        R=rotation,
        newCameraMatrix=matrix,
        size=(int(1280*SCALE), int(720*SCALE)),
        m1type=cv2.CV_32FC2
    )


maps = list(map(get_map, zip(matrices, rotations)))

# mapx_L, mapy_L = cv2.initUndistortRectifyMap(
#     cameraMatrix=params_L.camera_matrix,
#     distCoeffs=params_L.distortion_coefficients, R=R1,
#     newCameraMatrix=P1,
#     size=(int(1280*SCALE), int(720*SCALE)),
#     m1type=cv2.CV_32FC2
# )
mapx_L, mapy_L = maps[-1]

mapx_R, mapy_R = cv2.initUndistortRectifyMap(
    cameraMatrix=params_R.camera_matrix,
    distCoeffs=params_R.distortion_coefficients, R=R2,
    newCameraMatrix=P2,
    size=(int(1280*SCALE), int(720*SCALE)),
    m1type=cv2.CV_32FC2
)

names = ('R1', 'R2', 'P1', 'P2', 'Q', 'validPixROI1', 'validPixROI2')
for name in names:
    ll.debug('%s =\n%s', name, repr(globals()[name]))

ll.debug('ROI1 is width: %s, height: %s', *utils.width_height_from_roi(validPixROI1))
ll.debug('ROI2 is width: %s, height: %s', *utils.width_height_from_roi(validPixROI2))

# cap_L = cv2.VideoCapture('trabalho1/camera2.webm')
# cap_R = cv2.VideoCapture('trabalho1/camera1.webm')

cap_L, cap_R = map(cv2.VideoCapture, sources)

cap_L.set(cv2.CAP_PROP_POS_FRAMES, 20)
cap_R.set(cv2.CAP_PROP_POS_FRAMES, 20)

cv2.namedWindow('both')
# cv2.namedWindow('right')
cv2.moveWindow('both', 0, 0)
# cv2.resizeWindow('both', 1280, 720)

# cv2.moveWindow('right', 720, 0)

size = 0.56

points = []


def on_mouse(event, x, y, flags, userdata):  # NOQA
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        ll.info('click on %s', (event, x, y, flags, userdata))


cv2.setMouseCallback('both', on_mouse)  # NOQA


def draw_lines(frame, points):
    np.random.seed(0)
    for x, y in points:
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.line(frame, (0, y), (5000, y), color)
    return frame

def iterate(frame, maps, window):
    for mapx, mapy in itertools.cycle(maps):
        new_frame = cv2.resize(frame.copy(), dsize=None, fx=SCALE, fy=SCALE)
        new_frame = cv2.remap(new_frame, mapx, mapy, cv2.INTER_LANCZOS4)
        cv2.imshow(window, new_frame)
        key = cv2.waitKey(-1)
        if key & 0xFF == ord('q'):
            break

RECTIFY = False
# <editor-fold desc="Display">
while cap_L.isOpened() and cap_R.isOpened():
    success, frame_L = cap_L.read()
    if not success:
        break

    success, frame_R = cap_R.read()
    if not success:
        break

    if RECTIFY:
        frame_L = cv2.remap(frame_L, mapx_L, mapy_L, cv2.INTER_LANCZOS4)
        frame_R = cv2.remap(frame_R, mapx_R, mapy_R, cv2.INTER_LANCZOS4)
        # utils.rectangle_from_roi(frame_L, validPixROI1)
        # utils.rectangle_from_roi(frame_R, validPixROI2)

    both = cv2.hconcat((frame_L, frame_R))
    both = draw_lines(both, points)

    both = cv2.resize(both, dsize=(1280, 720), interpolation=cv2.INTER_LANCZOS4)
    cv2.imshow('both', both)
    key = cv2.waitKey(-1)

    if key != -1:
        key = key & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            RECTIFY = not RECTIFY  # toggle rectify ON or OFF
        elif key == ord('i'):
            iterate(frame_L, maps, 'left')
        elif key == ord('.'):
            cv2.imwrite('frame.png', both)

cap_L.release()
cap_R.release()
cv2.destroyAllWindows()
# </editor-fold>


def plot(opoints: np.ndarray):
    fig = pyplot.figure()
    ax = fig.add_subplot(111, projection='3d')

    def bloom_and_invert(vec):
        return vec[0][0], -1 * vec[1][0], -1 * vec[2][0]

    ax.scatter(opoints[..., 0], -1 * opoints[..., 1],
               -1 * opoints[..., 2], marker='o')

    abcd = string.ascii_lowercase[:4]
    for letter, point_xyz in zip(abcd, opoints.squeeze()):
        ll.debug(point_xyz)
        point_xyz[1] *= -1
        ax.text(*point_xyz, letter, size=20, zorder=1, color='k')

    R_L, _ = cv2.Rodrigues(rvec_L)
    R_R, _ = cv2.Rodrigues(rvec_R)
    C_L = np.dot(-R_L.transpose(), tvec_L)
    C_R = np.dot(-R_R.transpose(), tvec_R)

    def plot_cam(position: np.ndarray, label: str, marker="*"):
        ax.scatter(*bloom_and_invert(position), marker=marker)
        ax.text(*bloom_and_invert(position), label, size=20, zorder=1, color='k')

    # Plot Cams
    plot_cam(C_L, 'CAM 1', "^")
    plot_cam(C_R, 'CAM 2', "+")

    # rvec_compose, tvec_compose, *_ = cv2.composeRT(rvec_L, tvec_L, rvec_R, tvec_R)
    # R_compose, _ = cv2.Rodrigues(rvec_compose)
    # C_compose = np.dot(-R_compose.transpose(), tvec_compose)
    # plot_cam(C_compose, 'Cam 2 Compose', ".")

    pyplot.show()


if __name__ == '__main__':
    pass
    # plot(obj_points)
