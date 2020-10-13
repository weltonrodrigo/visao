# Busca de parâmetros para encontrar as flags adequadas no calibrateCamera
import glob
import itertools
from itertools import combinations

import cv2
import enlighten

from calibration import Calibration

cam = 2


def calibrate():
    images = glob.glob(
        f'/Users/torres/OneDrive/UNB/2020-08 Visão Computacional/'
        f'Trabalho 1/Calibration{cam}/*.jpg')

    calib = Calibration(delay=10, sample_size=10, cbcols=8, cbrows=6, alpha=1)

    for filename in images:
        img = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
        calib.add_sample(img)

    return calib


flags = (
    cv2.CALIB_FIX_K1,
    cv2.CALIB_FIX_K2,
    cv2.CALIB_FIX_K3,
    cv2.CALIB_FIX_K4,
    cv2.CALIB_FIX_K5,
    cv2.CALIB_FIX_K6,
    cv2.CALIB_ZERO_TANGENT_DIST,
)

names = (
    'K1',
    'K2',
    'K3',
    'K4',
    'K5',
    'K6',
)

dists = (None,)  # , *range(1, 7))


def sum2f(sigma):
    factors = []
    if sigma & cv2.CALIB_FIX_K1:
        factors.append('K1')
    if sigma & cv2.CALIB_FIX_K2:
        factors.append('K2')
    if sigma & cv2.CALIB_FIX_K3:
        factors.append('K3')
    if sigma & cv2.CALIB_FIX_K4:
        factors.append('K4')
    if sigma & cv2.CALIB_FIX_K5:
        factors.append('K5')
    if sigma & cv2.CALIB_FIX_K6:
        factors.append('K6')
    if sigma & cv2.CALIB_ZERO_TANGENT_DIST:
        factors.append('NT')

    return ";".join(factors)


if __name__ == "__main__":
    combs = map(
        lambda x: map(sum, combinations(flags, x + 1)), range(len(flags))
    )

    merged = list(itertools.chain.from_iterable(combs))

    calib = calibrate()

    how_many = len(dists) * len(merged)

    # pprint(list(merged))
    # pprint(list(map(sum2f, merged)))

    manager = enlighten.get_manager()

    # bingos = []
    # output = []
    # with manager.counter(total=how_many, desc='Basic', unit='ticks') as pbar:
    #     for dist in dists:
    #         for flags in merged:
    #             p = calib.calculate_camera_params(dist, flags)
    #             p = p.distortion_coefficients.flatten()
    #             output.append(f'{dist}-{sum2f(flags)}:\t\t\t{p}')
    #             if (p[2:] == 0).all():
    #                 bingos.append((dist, sum2f(flags), p))
    #                 # exit(0)
    #             pbar.update(1)

    # manager.stop()

    # print("\n".join(output))
    # print(list(map(lambda x: " ".join(x), bingos)), sep="\n")
    bingos = []
    output = []
    with manager.counter(total=how_many, desc='Basic', unit='ticks') as pbar:
        # for dist in dists:
        for flags in merged:
            error = calib.calculate_camera_params(dist, flags).total_error

            output.append(f'{error}: {calib.Calibration.flags}')
            if (p[2:] == 0).all():
                bingos.append((dist, sum2f(flags), p))
                # exit(0)
            pbar.update(1)

    manager.stop()

    print("\n".join(output))
    print(list(map(lambda x: " ".join(x), bingos)), sep="\n")
