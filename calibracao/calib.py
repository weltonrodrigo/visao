#!python
import glob
import logging
import sys
import time
from argparse import ArgumentParser

import cv2

from calibration import Calibration, CameraParams, ll as calibrationlog

logging.basicConfig(format="[%(levelname)s] %(asctime)s %(name)s %(message)s",
                    level=logging.INFO, stream=sys.stdout)
ll = logging.getLogger(__file__)

parser = ArgumentParser()
parser.add_argument('--alpha', default=0, type=float, help='alpha')
parser.add_argument('--size', default=0.5, type=float,
                    help='reduce image display')
parser.add_argument('--nodisplay', action='store_true',
                    help='do not show videos')
parser.add_argument('--write', metavar='FILE',
                    help='Write undistorted video to file')
parser.add_argument('--tangential', action='store_true',
                    help='enable tangential distortion factor')
parser.add_argument('--radial3', action='store_true',
                    help='Use three radial distortion factors')
parser.add_argument('--width', default=1280, type=int,
                    help='Width of the resulting image')
parser.add_argument('--height', default=720, type=int,
                    help='Height of the resulting image')
parser.add_argument('--read', action='store_true', help='file to use')
parser.add_argument('--nooutliers', default=False, action='store_true', help='remove outliers from calibration')
parser.add_argument('--verbose', '-v', action='store_true', help='Verbose')
parser.add_argument('cam', default=1, type=int, help='file to use')
parser.add_argument('file', nargs='?', default='/tmp/calibration.tmp', help='file to read from or write to')

res = parser.parse_args()

size = res.size

# flags = cv2.CALIB_FIX_PRINCIPAL_POINT + cv2.CALIB_FIX_ASPECT_RATIO
flags = 0
flags += 0 if res.tangential else cv2.CALIB_FIX_TANGENT_DIST
flags += 0 if res.radial3 else cv2.CALIB_FIX_K3

if res.verbose:
    # ll.setLevel(logging.DEBUG)
    # calibrationlog.setLevel(logging.DEBUG)
    logging.getLogger().setLevel(logging.DEBUG)

ll.info('Init')


def calibrate(alpha, flags, nooutliers=False):
    ll.info(__name__)
    images = glob.glob(
        f'/Users/torres/OneDrive/UNB/2020-08 Visão Computacional/'
        f'Trabalho 1/Calibration{res.cam}/*.jpg')

    images = sorted(images)

    calib = Calibration(delay=10, sample_size=10, cbcols=8, cbrows=6, alpha=alpha, nooutliers=nooutliers)

    ll.info('Made calib object')

    ll.debug(f'Reading {len(images)} images…')
    for filename in images:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        calib.add_sample(img, filename)

    params = calib.calculate_camera_params(flags=flags)

    if not params:
        ll.error('Erro ao calibrar.')

    ll.info(f'Writing camera params to file {res.file}')
    filenames = '\n'.join(images)
    comment = (
        f"With files:\n{filenames}\n"
        f'Camera: {res.cam}'
    )
    params.write_to_file(res.file, comment)

    return params


if res.write:
    output_video = cv2.VideoWriter(
        res.write,
        cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (res.width, res.height)
    )

if not res.read:
    start = time.time()
    params = calibrate(alpha=res.alpha, flags=flags, nooutliers=res.nooutliers)
    ll.info(f'Calibration took {time.time() - start}s')

if res.nodisplay and not res.write:
    ll.info('Done.')
    exit(0)

ll.info(f'Reading calibration from file {res.file}')
params = CameraParams.read_from_file(res.file)

# newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
#     params.camera_matrix, params.distortion_coefficients,
#     imageSize=(params.image_width, params.image_height),
#     newImgSize=(res.width, res.height), alpha=res.alpha)

newcameramtx, roi = Calibration.get_optimal_camera_matrix(
    params, new_image_size=(res.width, res.height), alpha=res.alpha
)
params.newcameramtx = newcameramtx

# mapx, mapy = cv2.initUndistortRectifyMap(
#     params.camera_matrix, params.distortion_coefficients,
#     newCameraMatrix=newcameramtx, size=(res.width, res.height),
#     m1type=cv2.CV_32FC2, R=None)

mapx, mapy = Calibration.get_undistort_maps(params)

# Convert Region of interest to int for Rectangle.
roi_x, roi_y, roi_w, roi_h = tuple(map(int, roi))

ll.debug(f"New camera matrix:\n{newcameramtx}")
ll.debug(f"ROI: {(roi_x, roi_y, roi_w, roi_h)}" )


cap = cv2.VideoCapture(
    f'/Users/torres/OneDrive/UNB/2020-08 Visão Computacional/'
    f'Trabalho 1/camera{res.cam}.webm')

if not res.nodisplay:
    WINDOW_TITLE = f"Video cam {res.cam}"

    cv2.namedWindow(WINDOW_TITLE)
    cv2.moveWindow(WINDOW_TITLE, x=0, y=0)

RECTIFY = True
if not res.nodisplay or res.write:
    while (cap.isOpened()):
        success, frame = cap.read()
        if not success:
            break

        if RECTIFY:
            frame = cv2.remap(frame, mapx, mapy, cv2.INTER_CUBIC)

        if res.write:
            output_video.write(frame)

        if not res.nodisplay:
            cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h),
                          (255, 0, 255), 3, 1)

            cv2.imshow(WINDOW_TITLE, frame)

            key = cv2.waitKey(10)
            if key and key & 0xFF == ord('q'):
                break
            if key and key & 0xFF == ord('r'):
                RECTIFY = False if RECTIFY == True else True
            if key and key & 0xFF == ord(' '):
                cv2.imwrite('frame.png', frame)
                cv2.waitKey(-1)  # wait until another key is pressed

if res.write:
    output_video.release()

cap.release()
cv2.destroyAllWindows()
