import numpy as np
import pptk

pcl = np.load('/home/patrik/mnt/rci/home/vacekpa2/lidar/data/datasets/nuscenes/pcl.npy')

# pcl_below = pcl[:,2] < -5
# pcl[:,2][pcl_below] = -pcl[:,2][pcl_below]
# pcl = pcl[pcl[:,2] > -1]
print(np.unique(pcl[:,3] / 255))
v=pptk.viewer(pcl[:,:3], (pcl[:,3] / 255) )
v.set(point_size=0.02)
