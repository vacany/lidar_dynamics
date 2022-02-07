import numpy as np
import os
import tools.ground_removal as gr

GRID_SIZE = 400


def get_voxels(number_of_frame, voxel_size=0.5):
    pts, _ = gr.get_frame_without_ground(number_of_frame)
    voxels = np.floor(pts / voxel_size).astype(int)
    return voxels


def get_voxels2d(number_of_frame, voxel_size=0.5):
    pts, _ = gr.get_frame_without_ground(number_of_frame)
    pts = pts[:, :2]
    voxels = np.floor(pts / voxel_size).astype(int)
    return voxels


def get_translator(voxels, points):
    """
    dict where keys are voxels and values are points inside voxel
    """
    translator = {}
    for i in range(voxels.shape[0]):
        voxel = translator.get(voxels[i].tobytes())
        if voxel is None:
            translator[voxels[i].tobytes()] = [points[i]]
        else:
            translator[voxels[i].tobytes()].append(points[i])
    return translator


def get_voxel_grid(voxel_size, first_frame, num_of_frames):
    """
    Create a N by N by T matrix of voxels, if a voxel X,Y at time T1 contains atleast
    one point from PCL, set matrix[X,Y,T1] = 1, loop through all time sequences and
    then sum across time, leaving one N by N matrix which at position X,Y contains the
    number of how many times from all times T did voxel[X,Y] contain points (number between 0 and T ->
    -> T if during each frame there were some points present at that given voxel)
    """

    grid_range = int(GRID_SIZE / voxel_size)
    voxel_grid = np.zeros((2 * grid_range, 2 * grid_range, num_of_frames), dtype=np.int8)
    for i in range(0, num_of_frames):
        pts, _ = gr.get_frame_without_ground(i + first_frame)
        v = get_voxels2d(i + first_frame, voxel_size) + grid_range  # point [0,0] will be in the center of the grid
        voxel_grid[v[:, 0], v[:, 1], i] = 1

    final_grid = np.sum(voxel_grid, axis=2)
    return final_grid


def get_dynamic_voxels(voxel_size, first_frame, num_of_frames):
    grid_range = int(GRID_SIZE / voxel_size)
    final_grid = get_voxel_grid(voxel_size, first_frame, num_of_frames)

    dynamic_coords = np.where(final_grid == 1)
    dynamic_voxels = np.zeros((dynamic_coords[0].shape[0], 2))
    dynamic_voxels[:, 0] = dynamic_coords[0] - grid_range  # return the points back from offset
    dynamic_voxels[:, 1] = dynamic_coords[1] - grid_range

    voxels_first = get_voxels2d(first_frame, voxel_size)
    voxels_last = get_voxels2d(first_frame + num_of_frames - 1, voxel_size)

    true_dynamic_voxels = []
    for dynamic_voxel in dynamic_voxels:
        if np.any(np.all(voxels_last == dynamic_voxel, axis=1)) or np.any(
                np.all(voxels_first == dynamic_voxel, axis=1)):
            continue
        else:
            true_dynamic_voxels.append(dynamic_voxel)

    true_dynamic_voxels = np.array(true_dynamic_voxels)
    return true_dynamic_voxels


def get_dynamic_points(voxel_size, first_frame, num_of_frames):
    dynamic_voxels = get_dynamic_voxels(voxel_size, first_frame, num_of_frames)
    dynamic_points = []
    # for i in range(1, num_of_time_sequences - 1): TODO change
    for i in range(first_frame, first_frame + num_of_frames):
        print(i)
        pts, _ = gr.get_frame_without_ground(i)
        v = get_voxels2d(i, voxel_size)
        for ii in range(len(v)):
            if np.any(np.all(v[ii] == dynamic_voxels, axis=1)):
                dynamic_points.append(pts[ii])
    dynamic_points = np.array(dynamic_points)
    return dynamic_points
