import os
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm


GROUND_IDX = 1
IGNORE_IDX = 0
FIRST_OBJECT_IDX = 2


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
SHOW = True
SEED = 25
NSHOW = 10

# apply RANSAC plane estimation
USE_RANSAC = True
GROUND_INLIER_THR = 0.1

# fill the holes in the ground estimation by looking at a neighboring labels (up and down)
FILL_INVALID_PTS_GROUND = True


''' Kitti - taken from Antonin Vobecky '''
def read_calib(calib_path):
    with open(calib_path, 'r') as f:
        calib = f.readlines()
    # P2 (3 x 4) for left eye
    P2 = np.matrix([float(x) for x in calib[2].strip('\n').split(' ')[1:]]).reshape(3, 4)
    # R0_rect = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 3)
    # Add a 1 in bottom-right, reshape to 4 x 4
    # R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0], axis=0)
    # R0_rect = np.insert(R0_rect, 3, values=[0, 0, 0, 1], axis=1)
    R0_rect = np.eye(4)
    # Tr_velo_to_cam = np.matrix([float(x) for x in calib[5].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.matrix([float(x) for x in calib[4].strip('\n').split(' ')[1:]]).reshape(3, 4)
    Tr_velo_to_cam = np.insert(Tr_velo_to_cam, 3, values=[0, 0, 0, 1], axis=0)
    return P2, R0_rect, Tr_velo_to_cam

def read_bin(bin_file):
    pointcloud = np.fromfile(bin_file, dtype=np.float32, count=-1).reshape([-1, 4])
    return pointcloud

def turn_ticks_off():
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

def project_pointcloud(cloud, image, P2, R0_rect, Tr_velo_to_cam, get_valid_pc=False):
    IMG_H, IMG_W, _ = image.shape

    valid = np.asarray(range(len(cloud)))

    velo = np.insert(cloud, 3, 1, axis=1).T
    behind_cam = np.where(velo[0, :] < 0)
    valid = np.delete(valid, behind_cam)
    velo = np.delete(velo, behind_cam, axis=1)
    pc_valid = np.delete(cloud, behind_cam, axis=0)

    # cam = np.asarray(P2 * R0_rect * Tr_velo_to_cam * velo)
    cam = np.asarray(P2 * Tr_velo_to_cam * velo)
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

if __name__ == '__main__':

    # start_idx, end_idx = int(sys.argv[1]), int(sys.argv[2])  # 0, 7482
    start_idx, end_idx = (400, 401)  # 0, 7482

    random.seed(SEED)
    data_root_dir = '/home/patrik/data/semantic-kitti/'

    split = '07'
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
        calib_path = os.path.join(split_data_dir, 'calib.txt')#, '{}.txt'.format(index))

        # read point cloud
        xyz = read_bin(bin_path)[:, :3]

        # read calibration
        P2, R0_rect, Tr_velo_to_cam = read_calib(calib_path)

        # read image
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        # read point cloud -> range image mapping
        # pc2ri_mapping = np.load(pc2ri_mapping_path)

        # project point cloud to the image, determine the points in the point cloud that project to the image
        cam, xyz_cam, valid_arr = project_pointcloud(xyz, image, P2, R0_rect, Tr_velo_to_cam, get_valid_pc=True)
        # pc2ri_mapping = pc2ri_mapping[valid_arr]
        # rmin, rmax = np.min(pc2ri_mapping[:, 0]), np.max(pc2ri_mapping[:, 0])
        # cmin, cmax = np.min(pc2ri_mapping[:, 1]), np.max(pc2ri_mapping[:, 1])

        ''' u, v coordinates of lidar in RGB image'''
        u, v, _ = cam
        u, v = np.asarray(u), np.asarray(v)

        IMG_H, IMG_W, _ = image.shape

        ind_u, ind_v = np.floor(v).astype('i4'), np.floor(u).astype('i4')

        # cam[2] = 1
        # cam = np.insert(cam, 3, 1, axis=0)
        # P2 = np.insert(P2, 3, values=[0, 0, 0, 1], axis=0)
        # out = np.linalg.inv(P2) @ cam

        # v = pptk.viewer(out)
        # v.set(point_size=0.03)

        colors_of_points = image[ind_u, ind_v] / 255
        # v = pptk.viewer(xyz_cam, colors_of_points)
        # v.set(point_size=0.03)
        #

        from data.point_clouds.bev import Bird_eye_view
        from data.trajectory.trajectory import line_traj

        traj = line_traj([[20,100], [190, 110]])
        points = np.concatenate((xyz_cam, colors_of_points), axis=1)
        bev_cls = Bird_eye_view(points, x_range=(0,50), y_range=(-25,25), cell_size=(0.25,0.25))

        bev_cls.rgb_map()
        bev = bev_cls.grid

        np.save('bev.npy', bev[...,:3].sum(2))
        # bev[traj[:,0], traj[:,1], 0] = 1
        # plt.imshow(bev[...,:3].sum(2))
        # plt.show()

