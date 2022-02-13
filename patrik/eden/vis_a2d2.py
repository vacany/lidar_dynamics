import numpy as np
import os
import glob
import sys

frame = int(sys.argv[1])

path = '/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/a2d2/camera_lidar/20180810_150607/lidar/'

lidar_paths = glob.glob(path + '*/*.npz')

frame_format = f"{frame:09d}.npz"

same_frame_data = [name for name in lidar_paths if frame_format in os.path.basename(name)]

pcl_list = []
for sweep in same_frame_data:
    file = np.load(sweep)
    pcl = np.concatenate((file['pcloud_points'], file['pcloud_attr.reflectance'][:,np.newaxis]), axis=1)

    pcl_list.append(pcl)

pcl = np.concatenate(pcl_list)

import pptk
v=pptk.viewer(pcl[:,:3], pcl[:,3])
v.set(point_size=0.02)
