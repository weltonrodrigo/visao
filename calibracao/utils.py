from typing import Tuple, List

import cv2


def rectangle_from_roi(frame, roi):
    """"Draws a rectangle for a given region of interest"""
    roi_x, roi_y, roi_w, roi_h = tuple(map(int, roi))
    cv2.rectangle(
        frame,
        (roi_x, roi_y),
        (roi_x + roi_w, roi_y + roi_h),
        (255, 0, 255), 3, 1)

def width_height_from_roi(roi: List[int]) -> Tuple[int, int]:
    """Returns width and height from a roi object"""
    roi_x, roi_y, roi_w, roi_h = map(int, roi)
    return roi_w, roi_h