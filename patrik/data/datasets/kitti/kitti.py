import numpy as np
import pickle
import glob

# Just for testing the detector
root = '/home/patrik/mnt/hdd/kitti/'

lidars = '/training/velodyne/'

with open(root + 'kitti_infos_trainval.pkl', 'rb') as f:
    train_data = pickle.load(f)

frame = 20
print(train_data[frame]['point_cloud'])
print(train_data[frame]['annos'].keys())

lidar_path = root + lidars + train_data[frame]['point_cloud']['lidar_idx'] + '.bin'
pcl = np.fromfile(lidar_path, dtype=np.float32).reshape(-1,4)

print(pcl.shape)

bbox = train_data[frame]['annos']['gt_boxes_lidar']
print(bbox)

from data.point_clouds.box import connect_3d_corners
corners = connect_3d_corners(bbox)
corners = np.insert(corners, 3, 1, axis=1)

pcl = np.concatenate((pcl, corners), axis=0)
import pptk
pptk.viewer(pcl[:,:3], pcl[:,3])
