import numpy as np


from data.datasets.semantic_kitti.semantic_kitti import SemKittiDataset
from exps.rw import load_yaml

from data.point_clouds.clustering import driving_scene


dataset = SemKittiDataset()



# import pptk
# pptk.viewer(batch['points'][:,:3], (batch['instance_label'] == 2) * (batch['seg_label'] == 2))

batch = dataset.get_multi_frame(0)
pcl1 = batch['points'][:,:3][(batch['instance_label'] == 2) * (batch['seg_label'] == 2)]
batch = dataset.get_multi_frame(5)
pcl2 = batch['points'][:,:3][(batch['instance_label'] == 2) * (batch['seg_label'] == 2)]

# import pptk
# pptk.viewer(pcl1)
# pptk.viewer(pcl2)

# subsample
pcl1 = pcl1[:-1,:]


from data.point_clouds.clustering.icp import icp

T, dist = icp(pcl1, pcl2)

import pptk
pcl1 = np.insert(pcl1, 3, 1, axis=1)
pcl2 = np.insert(pcl2, 3, 2, axis=1)

pcl3 = pcl1.copy()
pcl3[:,3] = 3
pcl3[:,:3] += T[:3,-1]

print(T)

all_pcl = np.concatenate((pcl1, pcl2, pcl3), axis=0)
v=pptk.viewer(all_pcl[:,:3], all_pcl[:,3])
v.set(point_size=0.02)
# render.Vis_function(dataset=dataset, function=clustering_annotator, config=config)
