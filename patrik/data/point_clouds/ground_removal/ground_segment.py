from scipy.signal import savgol_filter
from sklearn.cluster import DBSCAN

import numpy as np
from data.datasets.semantic_kitti.semantic_lasers import LaserScan

GROUND_IDX = 1
IGNORE_IDX = 0
FIRST_OBJECT_IDX = 2


MIN_ANGLE_DEG = -25.9
MIN_ANGLE_RAD = MIN_ANGLE_DEG * np.pi / 180.
MAX_ANGLE_RAD = 2.0 * np.pi / 180.


# parameter used as a threshold for deciding whether to separate two segments
THETA_DEG = 10.
THETA = THETA_DEG * np.pi / 180.

# maximal allowed angle difference during ground filtering
MAX_DIF_DEG = 5.
MAX_DIFF_GROUND = MAX_DIF_DEG * np.pi / 180.

# depth repair, number of steps
REPAIR_DEPTH_STEP = 5


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


def ground_segment_velodyne(pcl):
    laser = LaserScan(project=True)
    laser.open_scan(pcl)
    laser.do_range_projection()
    grid = laser.proj_range
    grid[grid == -1] = 0

    angles, grid = compute_angles(grid)

    grid = np.insert(grid, 63, -1, axis=0)

    # Smooth by columts
    for c in range(grid.shape[1]):
        grid[:, c] = savgol_filter(grid[:, c], 5, 2)

    repaired_image = repair_depth(grid, step=REPAIR_DEPTH_STEP, depth_threshold=1.0)

    ground_labels = label_ground(grid, repaired_image)

    pcl = laser.proj_xyz.reshape(-1, 3)
    inten = laser.proj_remission.reshape(-1)

    mask = (pcl != (-1, -1, -1)).all(1)
    pcl = pcl[mask]
    inten = inten[mask]

    pcl = np.concatenate((pcl, inten[:, None]), axis=1)

    labels = ground_labels.reshape(-1)
    labels = labels[mask]

    grid = grid.reshape(-1)
    grid = grid[mask]

    return pcl, labels, grid

def cluster_points(pcl, min_dist=0.4, min_samples=8):


    cluster_model = DBSCAN(min_dist, min_samples=min_samples)


    clusters = cluster_model.fit(pcl_non_ground)
    clustered_pcl = clusters.labels_

    return clustered_pcl

##################################### Ground removal

if __name__ == '__main__':
    pcl = np.fromfile('/home/patrik/data/semantic-kitti/04/velodyne/000000.bin', dtype=np.float32).reshape(-1, 4)
    labels = np.fromfile('/home/patrik/data/semantic-kitti/04/labels/000000.label', dtype=np.int16).reshape(-1,1)

    pcl, labels, angles = ground_segment_velodyne(pcl)
    pcl_non_ground = pcl[labels != 1]
    clustered_pcl = cluster_points(pcl_non_ground)

    import pptk
    # Projection and ground segmentation
    v = pptk.viewer(pcl, labels)
    v.set(point_size=0.03)

    # v2 = pptk.viewer(pcl[labels != 1], labels[labels != 1])
    # v2.set(point_size=0.03)

    # Clustering the rest of the point cloud
    # v3 = pptk.viewer(pcl_non_ground, clustered_pcl)
    # v3.set(point_size=0.03)
    # filtering the ground
    # v4 = pptk.viewer(pcl, angles)
    # v4.set(point_size=0.03)

    v5 = pptk.viewer(pcl_non_ground, pcl_non_ground[:,3] > 0.95)
    v5.set(point_size=0.03)

    # DSF Meeting
    #
