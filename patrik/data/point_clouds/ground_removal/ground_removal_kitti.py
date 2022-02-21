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

from tqdm import tqdm

from constants import GROUND_IDX, IGNORE_IDX, FIRST_OBJECT_IDX

BG_SHADE = 235
BG_COLOR = (BG_SHADE / 255, BG_SHADE / 255, BG_SHADE / 255)
FG_COLOR = (255 / 255, 127 / 255, 80 / 255)
IG_SHADE = 128
IGNORE_COLOR = (IG_SHADE / 255, IG_SHADE / 255, IG_SHADE / 255)

MIN_ANGLE_DEG = -24.9
MIN_ANGLE_RAD = MIN_ANGLE_DEG * np.pi / 180.
MAX_ANGLE_RAD = 2.0 * np.pi / 180.

VERT_ANG_RES_DEG = 0.4
VERT_ANG_RES_RAD = VERT_ANG_RES_DEG * np.pi / 180.
VERT_ANG_RES_COS, VERT_ANG_RES_SIN = np.cos(VERT_ANG_RES_RAD), np.sin(VERT_ANG_RES_RAD)

HOR_ANG_RES_DEG = 0.35
HOR_ANG_RES_RAD = HOR_ANG_RES_DEG * np.pi / 180.
HOR_ANG_RES_COS, HOR_ANG_RES_SIN = np.cos(HOR_ANG_RES_RAD), np.sin(HOR_ANG_RES_RAD)

# parameter used as a threshold for deciding whether to separate two segments
THETA_DEG = 10.
THETA = THETA_DEG * np.pi / 180.

# maximal allowed angle difference during ground filtering
MAX_DIF_DEG = 5.
MAX_DIFF_GROUND = MAX_DIF_DEG * np.pi / 180.

# depth repair, number of steps
REPAIR_DEPTH_STEP = 5

MIN_CLUSTER_SIZE = 10

# display params
PT_ALPHA = 0.5
PT_SIZE = 2.
FIGSIZE = (12, 4)
SHOW = False
SEED = 25
NSHOW = 10

# apply RANSAC plane estimation
USE_RANSAC = True
GROUND_INLIER_THR = 0.1

# fill the holes in the ground estimation by looking at a neighboring labels (up and down)
FILL_INVALID_PTS_GROUND = True

try:
    import open3d as o3d

    ver = o3d.__version__.split('.')
    if ver[0] != '0' or ver[1] != '9':
        print('You need to use 0.9 version of the Open3D!')
        USE_RANSAC = False
except:
    print('Cannot import open3d! Will not use the RANSAC refinement.')
    USE_RANSAC = False


def interpolate_range_to_image_v2(image, cp_points, ri_points, component_labels, cmap, ignore_idx=IGNORE_IDX,
                                  nuscenes=False, show=SHOW):
    # (x,y) style
    if nuscenes:
        if cp_points.shape[0] == 2 or cp_points.shape[0] == 3:
            cp_points = cp_points.T
        points = cp_points[:, :2].astype(int)
    else:
        points = cp_points[:, 1:3].astype(int)

    ri_rows, ri_cols = ri_points[:, 0], ri_points[:, 1]
    min_row, max_row, min_col, max_col = ri_rows.min(), ri_rows.max(), ri_cols.min(), ri_cols.max()
    cp_rmin_orig = int(max(v[ri_rows == min_row]))  # get MAX
    cp_cmin_orig = int(max(u[ri_cols == min_col]))  # get MAX
    cp_rmax_orig = int(min(v[ri_rows == max_row]))  # get MIN
    cp_cmax_orig = int(min(u[ri_cols == max_col]))

    # cp_cmin_orig, cp_cmax_orig = np.min(points[:, 0]), np.max(points[:, 0])
    # cp_rmin_orig, cp_rmax_orig = np.min(points[:, 1]), np.max(points[:, 1])

    # TODO: interpolate range image coordinates -> image coordinates mapping such that we have a mapping for all points in the range image
    values = points
    grid_x, grid_y = np.mgrid[0:component_labels.shape[1], 0:component_labels.shape[0]]
    start = time.time()

    # TODO: get the dense range image -> camera image mapping
    ri_cp_mapping = griddata(ri_points[:, [1, 0]], values, (grid_x, grid_y), method='linear').T
    valid = np.isfinite(ri_cp_mapping[0])

    # TODO: use the interpolated values
    ri_points_valid = np.stack((grid_y[valid.T].flatten(), grid_x[valid.T].flatten()))
    # where are the range image points mapped to in the image
    cp_points_interpolated = ri_cp_mapping[:, ri_points_valid[0], ri_points_valid[1]]

    # TODO: get the points that would be projected to reasonable locations
    valid_cols = np.bitwise_and(cp_points_interpolated[0] >= cp_cmin_orig, cp_points_interpolated[0] <= cp_cmax_orig)
    valid_rows = np.bitwise_and(cp_points_interpolated[1] >= cp_rmin_orig, cp_points_interpolated[1] <= cp_rmax_orig)
    valid_cp = np.bitwise_and(valid_cols, valid_rows)
    ri_points_valid = ri_points_valid[:, valid_cp]
    cp_points_interpolated = cp_points_interpolated[:, valid_cp].T

    points = cp_points_interpolated  # where I do have the values
    values = component_labels[ri_points_valid[0], ri_points_valid[1]]

    col_index, row_index = 0, 1

    rmin, rmax = int(cp_points_interpolated[:, row_index].min()), int(cp_points_interpolated[:, row_index].max())
    cmin, cmax = int(cp_points_interpolated[:, col_index].min()), int(cp_points_interpolated[:, col_index].max())

    grid_x, grid_y = np.mgrid[cmin:cmax, rmin:rmax]
    image_cropped = image[rmin:rmax + 1, cmin:cmax + 1]
    start = time.time()
    labels_interpolated = griddata(points, values, (grid_x, grid_y), method='nearest', fill_value=ignore_idx).T
    # print('interpolation in {:.1f}s'.format(time.time() - start))

    if show:
        start = time.time()
        tmp = labels_interpolated.copy()
        max_val = np.max(labels_interpolated)
        tmp[labels_interpolated == -1] = max_val + 1
        labels_interpolated_colored = (np.asarray(list(map(cmap, tmp))) * 255).astype(np.uint8)
        plt.figure()
        plt.imshow(labels_interpolated_colored)
        plt.show()

        plt.figure()
        plt.imshow(image_cropped)
        plt.imshow(labels_interpolated_colored, alpha=0.5)
        plt.show()
        # print('Show overlay in {:.1f}s'.format(time.time() - start))

    left, upper, right, lower = cmin, rmin, cmax, rmax
    return labels_interpolated, (left, upper, right, lower)


def turn_ticks_off():
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)


def generate_colormap(number_of_distinct_colors: int = 80):
    if number_of_distinct_colors == 0:
        number_of_distinct_colors = 80

    number_of_shades = 7
    number_of_distinct_colors_with_multiply_of_shades = int(
        math.ceil(number_of_distinct_colors / number_of_shades) * number_of_shades)

    # Create an array with uniformly drawn floats taken from <0, 1) partition
    linearly_distributed_nums = np.arange(
        number_of_distinct_colors_with_multiply_of_shades) / number_of_distinct_colors_with_multiply_of_shades
    # shift it a bit
    scale = 1.0
    shift = (1 - scale) / 2
    linearly_distributed_nums = linearly_distributed_nums * scale + shift

    # We are going to reorganise monotonically growing numbers in such way that there will be single array with saw-like pattern
    #     but each saw tooth is slightly higher than the one before
    # First divide linearly_distributed_nums into number_of_shades sub-arrays containing linearly distributed numbers
    arr_by_shade_rows = linearly_distributed_nums.reshape(number_of_shades,
                                                          number_of_distinct_colors_with_multiply_of_shades // number_of_shades)

    # Transpose the above matrix (columns become rows) - as a result each row contains saw tooth with values slightly higher than row above
    arr_by_shade_columns = arr_by_shade_rows.T

    # Keep number of saw teeth for later
    number_of_partitions = arr_by_shade_columns.shape[0]

    # Flatten the above matrix - join each row into single array
    nums_distributed_like_rising_saw = arr_by_shade_columns.reshape(-1)

    # HSV colour map is cyclic (https://matplotlib.org/tutorials/colors/colormaps.html#cyclic), we'll use this property
    initial_cm = hsv(nums_distributed_like_rising_saw)

    lower_partitions_half = number_of_partitions // 2
    upper_partitions_half = number_of_partitions - lower_partitions_half

    # Modify lower half in such way that colours towards beginning of partition are darker
    # First colours are affected more, colours closer to the middle are affected less
    lower_half = lower_partitions_half * number_of_shades
    if lower_half>0:
        for i in range(3):
            initial_cm[0:lower_half, i] *= np.arange(0.2, 1, 0.8 / lower_half)

    # Modify second half in such way that colours towards end of partition are less intense and brighter
    # Colours closer to the middle are affected less, colours closer to the end are affected more
    for i in range(3):
        for j in range(upper_partitions_half):
            modifier = np.ones(number_of_shades) - initial_cm[lower_half + j * number_of_shades: lower_half + (
                    j + 1) * number_of_shades, i]
            modifier = j * modifier / upper_partitions_half
            initial_cm[lower_half + j * number_of_shades: lower_half + (j + 1) * number_of_shades, i] += modifier

    return ListedColormap(initial_cm)


def compute_angles(range_image):
    # go column by column

    min_angle = MIN_ANGLE_RAD
    max_angle = MAX_ANGLE_RAD
    vertical_resolution = range_image.shape[0]

    vertical_angles = np.linspace(min_angle, max_angle, vertical_resolution)
    sines = np.sin(vertical_angles)
    cosines = np.cos(vertical_angles)

    angles = np.zeros((range_image.shape[0] - 1, range_image.shape[1]))
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


def smooth_golay(column_angles, window_size=5, polyorder=2):
    '''

    :param angles: AxBx3 matrix of precomputed angles between points
    :return: smoothed angles
    '''

    # TODO: perform the smoothing for each column separately
    smoothed = savgol_filter(column_angles, window_length=window_size, polyorder=polyorder)
    return smoothed


def read_bin(bin_file):
    pointcloud = np.fromfile(bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
    return pointcloud


def getGround_World(points, THRESHOLD):
    '''
    filter ground and non_ground points
    :param points array
    :param threshold value (varies between 0.1 to 0.3 generally)
    '''
    xyz = points
    height_col = int(np.argmin(np.var(xyz, axis=0)))

    temp = np.zeros((len(xyz[:, 1]), 4), dtype=float)
    temp[:, :3] = xyz[:, :3]
    temp[:, 3] = np.arange(len(xyz[:, 1]))
    xyz = temp
    z_filter = xyz[(xyz[:, height_col] < np.mean(xyz[:, height_col]) + 1.5 * np.std(xyz[:, height_col])) & (
            xyz[:, height_col] > np.mean(xyz[:, height_col]) - 1.5 * np.std(xyz[:, height_col]))]

    max_z, min_z = np.max(z_filter[:, height_col]), np.min(z_filter[:, height_col])
    z_filter[:, height_col] = (z_filter[:, height_col] - min_z) / (max_z - min_z)
    iter_cycle = 10
    for i in range(iter_cycle):
        covariance = np.cov(z_filter[:, :3].T)
    w, v, h = np.linalg.svd(np.matrix(covariance))
    normal_vector = w[np.argmin(v)]
    filter_mask = np.asarray(np.abs(np.matrix(normal_vector) * np.matrix(z_filter[:, :3]).T) < THRESHOLD)
    z_filter = np.asarray([z_filter[index[1]] for index, a in np.ndenumerate(filter_mask) if a == True])

    z_filter[:, height_col] = z_filter[:, height_col] * (max_z - min_z) + min_z
    world = np.array([row for row in xyz if row[3] not in z_filter[:, 3]])

    return z_filter, world


def project_pointcloud(cloud, image, P2, R0_rect, Tr_velo_to_cam, get_valid_pc=False):
    IMG_H, IMG_W, _ = image.shape

    valid = np.asarray(range(len(cloud)))

    velo = np.insert(cloud, 3, 1, axis=1).T
    behind_cam = np.where(velo[0, :] < 0)
    valid = np.delete(valid, behind_cam)
    velo = np.delete(velo, behind_cam, axis=1)
    pc_valid = np.delete(cloud, behind_cam, axis=0)

    cam = np.asarray(P2 * R0_rect * Tr_velo_to_cam * velo)
    behind_cam = np.where(cam[2] < 0)
    valid = np.delete(valid, behind_cam)
    cam = np.delete(cam, behind_cam, axis=1)
    pc_valid = np.delete(pc_valid, behind_cam, axis=0)

    # get u,v,z
    cam[:2] /= cam[2, :]
    # do projection stuff
    # filter point out of canvas
    u, v, z = cam
    u, v = np.asarray(u).flatten(), np.asarray(v).flatten()
    u_out = np.logical_or(u < 0, u > IMG_W)
    v_out = np.logical_or(v < 0, v > IMG_H)
    outlier = np.logical_or(u_out, v_out)
    valid = np.delete(valid, np.where(outlier))
    cam = np.delete(cam, np.where(outlier), axis=1)
    pc_valid = np.delete(pc_valid, np.where(outlier), axis=0)

    if get_valid_pc:
        return cam, pc_valid, valid
    else:
        return cam


def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    return P2, R0_rect, Tr_velo_to_cam


def deg2rad(deg):
    return deg * np.pi / 180.


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


def neighborhood_w_direction(r, c, valid, rmax, cmax):
    rmax, cmax = rmax - 1, cmax - 1
    V, H = 'vertical', 'horizontal'
    directions = [(-1, 0, V), (1, 0, V), (0, -1, H), (0, 1, H)]
    neighbors = []

    for dr, dc, drc in directions:
        nr, nc = r + dr, c + dc
        if nr < 0 or nr > rmax or nc < 0 or nc > cmax:
            continue
            neighbors.append((nr, nc, drc))
        neighbors.append((nr, nc, drc))

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


def resize(arr, new_h, new_w):
    W, H = arr.shape[:2]
    xrange = lambda x: np.linspace(0, 1, x)
    f = interpolate.interp2d(xrange(H), xrange(W), arr, kind="linear")
    new_arr = f(xrange(new_w), xrange(new_h))
    return new_arr


def resize_nearest(arr, new_h, new_w):
    W, H = arr.shape[:2]
    xrange = lambda x: np.linspace(0, 1, x)
    f = interpolate.interp2d(xrange(H), xrange(W), arr, kind="nearest")
    new_arr = f(xrange(new_w), xrange(new_h))
    return new_arr


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


def repair_depth_my(range_image, step=REPAIR_DEPTH_STEP, depth_threshold=1.0):
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
                        prev = inpainted_depth[r - i, c]
                        nxt = inpainted_depth[r + j, c]
                        if prev > 0.001 and nxt > 0.001 and np.abs(prev - nxt) < depth_threshold:
                            _sum += (prev + nxt)
                            counter += 2
                if counter > 0:
                    curr_depth = _sum / counter
                    inpainted_depth[r, c] = curr_depth
    return inpainted_depth


def label_range_image(range_image, L, min_cluster_size=MIN_CLUSTER_SIZE, valid_measurement=None):
    label = FIRST_OBJECT_IDX
    nrows, ncols = range_image.shape

    for r in range(nrows):
        for c in range(ncols):
            if L[r, c] == 0 and range_image[r, c] > 0.001:
                if (valid_measurement is not None and not valid_measurement[r, c]):
                    # this is not a valid measurement
                    continue
                # not labeled and valid
                L, valid_component = label_component_bfs(r, c, label, L, range_image,
                                                         valid_measurement=valid_measurement,
                                                         min_cluster_size=min_cluster_size)
                if valid_component:
                    # valid, we can move on to the other label
                    label += 1
                else:
                    # invalid, too few points
                    L[L == label] = IGNORE_IDX

    return L


def get_laser_angle(row, col, direction):
    if direction == 'vertical':
        return VERT_ANG_RES_COS, VERT_ANG_RES_SIN
    elif direction == 'horizontal':
        return HOR_ANG_RES_COS, HOR_ANG_RES_SIN
    else:
        raise Exception('Unknown direction "{}".'.format(direction))


def label_component_bfs(r, c, label, L, range_image, valid_measurement=None, min_cluster_size=MIN_CLUSTER_SIZE):
    queue = {(r, c)}

    if valid_measurement is None:
        valid_measurement = range_image > 0.001

    nlabeled = 0

    while queue:
        r, c = queue.pop()
        current_label = L[r, c]
        if current_label > 0:
            # already labeled
            continue

        current_depth = range_image[r, c]
        cur_valid = valid_measurement[r, c]
        if current_depth < 0.001 or not cur_valid:
            # invalid point
            continue

        L[r, c] = label
        nlabeled += 1

        for rn, rc, direction in neighborhood_w_direction(r, c, range_image, L.shape[0] - 1, L.shape[1] - 1):

            d1 = max(range_image[r, c], range_image[rn, rc])
            d2 = min(range_image[r, c], range_image[rn, rc])
            cos_psi, sin_psi = get_laser_angle(r, c, direction)
            num = d2 * sin_psi
            den = d1 - d2 * cos_psi
            angle = np.arctan2(num, den)
            if angle > THETA:
                queue.add((rn, rc))

    valid = nlabeled > min_cluster_size
    return L, valid


def vec2homo(vec):
    dim = vec.shape[1]
    homo = np.insert(vec, dim, 1, axis=1)
    return homo


def distance2plane(plane, points):
    '''

    :param plane: equation of a plane, given by four numbers
    :param points: Nx3 numpy array
    :return: distance of each point to the plane
    '''
    if points.shape[1] == 3:
        points = vec2homo(points)
    num = np.abs(np.matmul(points, plane))
    den = np.sqrt(np.matmul(plane[:3], plane[:3]))
    dist = num / den
    return dist


def ransac(pcd, distance_thr=0.1, ransac_n=3, num_iterations=3000):
    '''

    :param pcd: point cloud
    :param distance_thr: distance threshold from the plane to consider a point inlier or outlier
    :param ransac_n: the number of sampled points drawn (3 as we want a plane)
    :param num_iterations: number of iterations
    :return:
    '''
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_thr, ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    # print(inliers)
    inlier_cloud = pcd.select_down_sample(inliers)
    outlier_cloud = pcd.select_down_sample(inliers, invert=True)
    return plane_model, inliers, inlier_cloud, outlier_cloud


def npy2pcd(arr):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)
    return pcd


def ransac_refinement(ground_labels, xyz_cam, mapping_shifted, u, v, image, return_plane=False):
    inverse_mapping = collections.defaultdict(list)
    for i in range(len(mapping_shifted)):
        inverse_mapping[tuple(mapping_shifted[i])].append(i)
    ground_pts_idx = np.asarray(np.where(ground_labels)).T
    tmp = []
    for idx in ground_pts_idx:
        key = tuple(idx)
        if key in inverse_mapping.keys():
            tmp.extend(inverse_mapping[key])
    ground_pts_idx = tmp
    ground_pts = xyz_cam[ground_pts_idx]
    plane_model, ground_labels_ransac, inlier_cloud, outlier_cloud = ransac(npy2pcd(ground_pts),
                                                                            distance_thr=GROUND_INLIER_THR)
    # TODO: compute distances to the plane
    dist2plane_all = distance2plane(plane_model, xyz_cam)
    # inliers
    inliers_idx = np.where(dist2plane_all <= GROUND_INLIER_THR)[0]
    ground_pts_new = xyz_cam[inliers_idx]
    # TODO: use those inliers as new ground estimation, only append new ground labels (do not delete the old ones)
    # ground_labels = np.zeros_like(ground_labels, dtype=bool)
    ground_labels_ransac_mapped = mapping_shifted[inliers_idx]
    ground_labels[ground_labels_ransac_mapped[:, 0], ground_labels_ransac_mapped[:, 1]] = True
    if SHOW:
        # TODO: show difference in ground estimation
        prev, new = set(ground_pts_idx), set(inliers_idx)
        overlapping = [x for x in prev if x in new]
        removed = [x for x in prev if x not in new]
        added = [x for x in new if x not in prev]
        u_over, v_over = u[overlapping], v[overlapping]
        u_rem, v_rem = u[removed], v[removed]
        u_add, v_add = u[added], v[added]

        plt.figure(figsize=FIGSIZE, dpi=96, tight_layout=True)
        # restrict canvas in range
        IMG_H, IMG_W = image.shape[:2]
        plt.axis([0, IMG_W, IMG_H, 0])
        plt.imshow(image)
        plt.scatter([u_over], [v_over], c='b', alpha=PT_ALPHA, s=PT_SIZE, label='overlapping')
        plt.scatter([u_rem], [v_rem], c='r', alpha=PT_ALPHA, s=PT_SIZE, label='removed')
        plt.scatter([u_add], [v_add], c='g', alpha=PT_ALPHA, s=PT_SIZE, label='added')
        plt.legend()
        plt.title('comparison')
        turn_ticks_off()
        plt.tight_layout()
        plt.show()
    if return_plane:
        return ground_labels, dist2plane_all, plane_model
    else:
        return ground_labels


def label_ground(angles_filtered, range_image_repaired):
    ground_labels = np.zeros_like(angles_filtered).astype(dtype=bool)
    max_start_rad = deg2rad(45.)
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


if __name__ == '__main__':

    # start_idx, end_idx = int(sys.argv[1]), int(sys.argv[2])  # 0, 7482
    start_idx, end_idx = (2, 3)  # 0, 7482

    random.seed(SEED)
    data_root_dir = './data'
    # data_root_dir = '/nfs/datasets/kitti'
    split = 'training'
    range_images_dir = 'range_images_depth_v2'
    split_data_dir = os.path.join(data_root_dir, split)
    data_save_dir = os.path.join(split_data_dir, 'data_w_labels')
    if not os.path.exists(data_save_dir):
        os.makedirs(data_save_dir)

    indices = list(range(start_idx, end_idx))

    for index in tqdm(indices, total=len(indices)):
        index = str(index).zfill(6)

        # path to the range image of the scene, range image contains raw lidar readings
        range_img_path = os.path.join(split_data_dir, range_images_dir, '{}.npy'.format(index))
        # path to the mapping from the 3D point cloud to the range image
        pc2ri_mapping_path = os.path.join(split_data_dir, range_images_dir, '{}_valid.npy'.format(index))
        # path to the image
        img_path = os.path.join(split_data_dir, 'image_2', '{}.png'.format(index))
        # path to the point cloud
        bin_path = os.path.join(split_data_dir, 'velodyne', '{}.bin'.format(index))
        # path to the image calibration
        calib_path = os.path.join(split_data_dir, 'calib', '{}.txt'.format(index))

        # read point cloud
        xyz = read_bin(bin_path)[:, :3]

        # read calibration
        P2, R0_rect, Tr_velo_to_cam = read_calib(calib_path)

        # read image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # read point cloud -> range image mapping
        pc2ri_mapping = np.load(pc2ri_mapping_path)



        # project point cloud to the image, determine the points in the point cloud that project to the image
        cam, xyz_cam, valid_arr = project_pointcloud(xyz, image, P2, R0_rect, Tr_velo_to_cam, get_valid_pc=True)
        pc2ri_mapping = pc2ri_mapping[valid_arr]
        rmin, rmax = np.min(pc2ri_mapping[:, 0]), np.max(pc2ri_mapping[:, 0])
        cmin, cmax = np.min(pc2ri_mapping[:, 1]), np.max(pc2ri_mapping[:, 1])

        ''' u, v coordinates of lidar in RGB image'''
        u, v, _ = cam
        u, v = np.asarray(u), np.asarray(v)


        IMG_H, IMG_W, _ = image.shape


        # load range image
        range_image = np.load(range_img_path)
        range_image = range_image[rmin:(rmax + 1), cmin:(cmax + 1)]

        # repair the range image as there might be some rows/columns missing
        range_image_repaired = repair_depth(range_image)

        # compute the angles
        angles, angles_filtered = compute_angles(range_image_repaired)

        # shift mapping
        # this mapping is from point cloud to range image (range image coords are [row, column])
        pc2ri_mapping_shifted = pc2ri_mapping - [rmin, cmin]
        pc2ri_mapping_shifted[:, 0] = np.clip(pc2ri_mapping_shifted[:, 0], 0, angles_filtered.shape[0] - 1)
        pc2ri_mapping_shifted[:, 1] = np.clip(pc2ri_mapping_shifted[:, 1], 0, angles_filtered.shape[1] - 1)


        # label ground
        ground_labels = label_ground(angles_filtered, range_image_repaired)

        # fill the holes by using window max pooling
        if FILL_INVALID_PTS_GROUND:
            invalid = np.where(range_image[:-1] < 0.001)
            ground_labels = fill_invalid_pts_ground(ground_labels, invalid)

        if USE_RANSAC:
            # refine the ground plane using RANSAC
            ground_labels, dist2plane_all, plane_model = ransac_refinement(ground_labels, xyz_cam,
                                                                           pc2ri_mapping_shifted, u,
                                                                           v, image, return_plane=True)

        # label the rest of the point cloud/range image
        range_image_wo_ground = range_image_repaired[:-1].copy()
        range_image_wo_ground[ground_labels] = 0.

        component_labels = np.zeros_like(range_image_wo_ground, dtype=int)
        component_labels[ground_labels] = GROUND_IDX
        invalid = (range_image < 0.001)[:-1]
        valid_measurement = None  # (range_image > 0.001)[:-1]
        component_labels = label_range_image(range_image_wo_ground, component_labels,
                                             valid_measurement=valid_measurement)
        # component_labels_all = component_labels.copy()
        # component_labels[invalid] = IGNORE_IDX

        nlabels = len(np.unique(component_labels))
        cmap = generate_colormap(nlabels)
        if SHOW:
            print('{} components'.format(nlabels))
            plt.imshow(cv2.resize(component_labels, dsize=(2048, 512), interpolation=cv2.INTER_NEAREST),
                       cmap=cmap)
            plt.imshow(component_labels, cmap=cmap)
            plt.title('components')
            turn_ticks_off()
            plt.tight_layout()
            plt.show()

        # colorize the visible point cloud
        xyz_cam_colorized = component_labels[pc2ri_mapping_shifted[:, 0], pc2ri_mapping_shifted[:, 1]]
        # xyz_cam_colorized_all = component_labels_all[pc2ri_mapping_shifted[:, 0], pc2ri_mapping_shifted[:, 1]]

        # project labels to image
        labels_in_image, crop_params = interpolate_range_to_image_v2(image, cam[:2], pc2ri_mapping_shifted,
                                                                     component_labels,
                                                                     cmap, nuscenes=True, show=SHOW)
        # labels_in_image_all, top_row_all = interpolate_range_to_image_v2(image, cam[:2], pc2ri_mapping_shifted,
        #                                                                  component_labels_all, cmap, nuscenes=True,
        #                                                                  show=SHOW)

        data = {
            'range_img_path': range_img_path,
            'ri_rows': (rmin, rmax + 1),
            'ri_cols': (cmin, cmax + 1),
            'calib_path': calib_path,
            'image_path': img_path,
            'ri_points_cam': pc2ri_mapping_shifted,
            'cp_points_cam': cam[:2],
            'pc_cam': xyz_cam,
            'ri_labelled': component_labels,
            # 'ri_labelled_all': component_labels_all,
            'labels_pc_cam': xyz_cam_colorized,
            # 'labels_pc_cam_all': xyz_cam_colorized_all,
            'labels_in_image': labels_in_image,  # labels projected to the image
            # 'labels_in_image_all': labels_in_image_all,
            'crop_params': crop_params,
            # 'labels_in_image_top_row_all': top_row_all
        }
        if USE_RANSAC:
            data['dist_to_plane'] = dist2plane_all
            data['plane_params'] = plane_model

        save_file_path = os.path.join(data_save_dir, '{}.pkl'.format(index))
        with open(save_file_path, 'wb') as f:
            pickle.dump(data, f)

        ground_idx = np.where((xyz_cam_colorized == GROUND_IDX))
        ignore_idx = np.where((xyz_cam_colorized == IGNORE_IDX))
        ground_and_ignore_idx = np.concatenate((ground_idx[0], ignore_idx[0]))
        objects_idx = np.delete(np.asarray(range(len(xyz_cam_colorized))), ground_and_ignore_idx)

        # objects
        v_obj, u_obj, c_obj = v[objects_idx], u[objects_idx], xyz_cam_colorized[objects_idx]

        # ground
        v_ground, u_ground, c_ground = v[ground_idx], u[ground_idx], BG_COLOR

        # ignore
        v_ig, u_ig, c_ig = v[ignore_idx], u[ignore_idx], IGNORE_COLOR

        assert len(v_ig) + len(v_obj) + len(v_ground) == len(v)



        import pptk

        ind_u, ind_v = np.floor(v).astype('i4'), np.floor(u).astype('i4')

        colors_of_points = image[ind_u, ind_v] / 255


        v = pptk.viewer(xyz_cam, xyz_cam_colorized == GROUND_IDX)
        v.set(point_size=0.03)
