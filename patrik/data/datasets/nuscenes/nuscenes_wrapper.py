from nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud

import socket
host = socket.gethostname()
if host == "Patrik":
    dataroot = '/home/patrik/mnt/rci/mnt/beegfs/gpu/temporary/vacekpa2/nuscenes/data/sets'
else:
    dataroot = '/mnt/beegfs/gpu/temporary/vacekpa2/nuscenes/data/sets'

nusc = NuScenes(version='v1.0-mini', dataroot=dataroot, verbose=True)

my_sample = nusc.sample[400]

sample_data_token = my_sample['data']['LIDAR_TOP']
sd_record = nusc.get('sample_data', sample_data_token)
sample_rec = nusc.get('sample', sd_record['sample_token'])
chan = 'LIDAR_TOP'
ref_chan = 'LIDAR_TOP'
nsweeps=20

pc, times = LidarPointCloud.from_file_multisweep(nusc=nusc,
                                                 sample_rec=sample_rec, chan=chan, ref_chan=ref_chan, nsweeps=nsweeps)
pcl = pc.points.T
times = times[0].T

# import matplotlib
# matplotlib.use('TkAgg')
# nusc.render_sample(my_sample['token'], show_lidarseg=True, filter_lidarseg_labels=[2,3,4])


import numpy as np
pcl = np.concatenate((pcl, times[:, np.newaxis]), axis=1)
np.save('pcl.npy', pcl)
