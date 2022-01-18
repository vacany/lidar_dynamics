import collections
import os
import random
import pickle
import sys
import time

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.interpolate import griddata
from scipy.signal import savgol_filter
from scipy import interpolate
import math
import numpy as np

from matplotlib.cm import hsv

MIN_ANGLE_DEG = -24.9
MIN_ANGLE_RAD = MIN_ANGLE_DEG * np.pi / 180.
MAX_ANGLE_RAD = 2.0 * np.pi / 180.

VERT_ANG_RES_DEG = 0.4
VERT_ANG_RES_RAD = VERT_ANG_RES_DEG * np.pi / 180.
VERT_ANG_RES_COS, VERT_ANG_RES_SIN = np.cos(VERT_ANG_RES_RAD), np.sin(VERT_ANG_RES_RAD)

HOR_ANG_RES_DEG = 0.35
HOR_ANG_RES_RAD = HOR_ANG_RES_DEG * np.pi / 180.
HOR_ANG_RES_COS, HOR_ANG_RES_SIN = np.cos(HOR_ANG_RES_RAD), np.sin(HOR_ANG_RES_RAD)

MAX_DIF_DEG = 5.
MAX_DIFF_GROUND = MAX_DIF_DEG * np.pi / 180.

REPAIR_DEPTH_STEP = 5


def smooth_golay(column_angles, window_size=5, polyorder=2):
    '''

    :param angles: AxBx3 matrix of precomputed angles between points
    :return: smoothed angles
    '''

    # TODO: perform the smoothing for each column separately
    smoothed = savgol_filter(column_angles, window_length=window_size, polyorder=polyorder)
    return smoothed


def compute_angles(range_image):
    # go column by column

    min_angle = MIN_ANGLE_RAD
    max_angle = MAX_ANGLE_RAD
    vertical_resolution = range_image.shape[0]

    vertical_angles = np.linspace(min_angle, max_angle, vertical_resolution)
    sines = np.sin(vertical_angles)
    cosines = np.cos(vertical_angles)

    angles = np.zeros(
        (range_image.shape[0] - 1, range_image.shape[1]))  # -1 because we compute angle differences between pairs
    angles_filtered = np.zeros((range_image.shape[0] - 1, range_image.shape[1]))

    for col_idx in range(range_image.shape[1]):
        col = range_image[:, col_idx][::-1]

        # start from the bottom, proceed upwards
        dz = np.abs(col[:-1] * sines[:-1] - col[1:] * sines[1:])
        dx = np.abs(col[:-1] * cosines[:-1] - col[1:] * cosines[1:])
        alpha = np.arctan2(dz, dx)[::-1]
        alpha_smoothed = smooth_golay(alpha)

        angles[:, col_idx] = alpha
        angles_filtered[:, col_idx] = alpha_smoothed

    return angles, angles_filtered


def neighborhood(r, c, valid, rmax, cmax):
    rmax, cmax = rmax - 1, cmax - 1
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    neighbors = []

    for dr, dc in directions:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr > rmax or nc < 0 or nc > cmax:
            continue
            neighbors.append((nr, nc))
        neighbors.append((nr, nc))

    return neighbors


def labelGroundBFS(r, c, angles, depths, labels, max_diff):
    queue = [(r, c)]
    visited = set()
    while queue:
        r, c = queue.pop(0)
        visited.add((r, c))
        if labels[r, c]:
            # already visited
            continue

        if angles[r, c] < 0.001:
            continue

        labels[r, c] = True

        for rn, cn in neighborhood(r, c, depths, *angles.shape):
            if (rn, cn) in visited or (rn, cn) in queue:
                continue
            diff = np.abs(angles[r, c] - angles[rn, cn])
            if diff < max_diff:
                queue.append((rn, cn))
    return labels


def repair_depth(range_image, step=REPAIR_DEPTH_STEP, depth_threshold=1.0):
    inpainted_depth = range_image.copy()
    rows, cols = range_image.shape
    for c in range(cols):
        for r in range(rows):
            curr_depth = inpainted_depth[r, c]
            if curr_depth < 0.001:
                counter = 0
                _sum = 0.
                for i in range(1, step + 1):
                    if (r - i) < 0:
                        continue
                    for j in range(1, step + 1):
                        if (r + j > (rows - 1)):
                            continue
                        prev = inpainted_depth[r - 1, c]
                        nxt = inpainted_depth[r + 1, c]
                        if prev > 0.001 and nxt > 0.001 and np.abs(prev - nxt) < depth_threshold:
                            _sum += (prev + nxt)
                            counter += 2
                if counter > 0:
                    curr_depth = _sum / counter
                    inpainted_depth[r, c] = curr_depth
    return inpainted_depth


def label_ground(angles_filtered, range_image_repaired):
    ground_labels = np.zeros_like(angles_filtered).astype(dtype=bool)
    max_start_rad = np.deg2rad(45.)
    last_row = angles_filtered.shape[0] - 1
    for c in range(angles_filtered.shape[-1]):
        r = last_row
        while r > 0 and range_image_repaired[r, c] < 0.001:
            r -= 1
        current_label = ground_labels[r, c]
        if current_label:
            # already labeled, skip
            continue
        if angles_filtered[r, c] >= max_start_rad:
            # too large angle, skip
            # continue
            pass
        ground_labels = labelGroundBFS(r, c, angles_filtered, range_image_repaired, ground_labels, MAX_DIFF_GROUND)

    return ground_labels


def fill_invalid_pts_ground(ground_labels, invalid_pts):
    ground_labels_filled = ground_labels.copy()
    h, w = ground_labels.shape
    invalid_rows, invalid_cols = invalid_pts

    for r, c in zip(invalid_rows, invalid_cols):
        if r > 0 and r < (h - 1):
            ground_labels_filled[r, c] = ground_labels_filled[r, c] or ground_labels[r - 1, c] and ground_labels[
                r + 1, c]

    return ground_labels_filled
